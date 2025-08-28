import asyncio
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace
from typing import List, Tuple, Optional

import pytest
import torch
from unittest.mock import Mock, AsyncMock, patch, call

# Under test
from myserve.scheduler import (
    GenRequest,
    Scheduler,
    _handle_prefill_req,
    _handle_prefill_batch,
    _handle_decode_batch,
    _handle_out,
)
from myserve.core.sampling import SamplerCfg


# ---------------------------
# Test helpers & fakes
# ---------------------------

class TinyTok:
    """A tiny tokenizer sufficient for unit tests.

    - Tokenizes by simple whitespace split
    - Provides pad/eos ids
    - decode() makes a readable token string like "T17"
    """

    def __init__(self, pad_token_id: int = 0, eos_token_id: int = 2, vocab_size: int = 50):
        self.pad_token_id = pad_token_id
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size

    def __call__(self, text, return_tensors="pt", add_special_tokens=False):
        if isinstance(text, str):
            words = text.split()
            # simple deterministic mapping per word length
            toks = [10 + (len(w) % (self.vocab_size - 10)) for w in words]
            if not toks:
                toks = [11]
            ids = torch.tensor([toks], dtype=torch.long)
            return {"input_ids": ids}
        elif isinstance(text, list):
            batch = []
            for t in text:
                words = t.split()
                toks = [10 + (len(w) % (self.vocab_size - 10)) for w in words]
                if not toks:
                    toks = [11]
                batch.append(torch.tensor(toks, dtype=torch.long))
            # left pad to max length
            L = max(x.numel() for x in batch)
            out = []
            for x in batch:
                pad = L - x.numel()
                out.append(torch.cat([torch.full((pad,), self.pad_token_id), x]))
            ids = torch.stack(out).long()
            return {"input_ids": ids}
        else:
            raise TypeError("Unsupported input to TinyTok")

    def decode(self, ids, skip_special_tokens=True, clean_up_tokenization_spaces=False):
        if torch.is_tensor(ids):
            ids = ids.tolist()
        if isinstance(ids, list):
            # take the last id for a readable piece
            return f"T{ids[-1]}"
        return f"T{int(ids)}"


@dataclass
class FakeBundle:
    device: torch.device
    model: any


class FakeModel:
    """A deterministic fake model that:
    - Returns logits with a sharp peak at (sum(visible inputs) % V)
    - Returns past with shapes [B, H, Tm, D] for L layers (Tm = max seq len for the batch),
      where the *last* t_new[b] positions are "real" and leading positions are zero-padded.
    Deterministic per-input so batched vs. serial runs match.
    """

    def __init__(self, vocab_size=50, num_layers=3, num_heads=2, head_dim=4):
        self.vocab_size = vocab_size
        self.L = num_layers
        self.H = num_heads
        self.D = head_dim
        self.calls: List[dict] = []  # record last calls for assertions
        self.config = SimpleNamespace(
            num_hidden_layers=num_layers,
            num_attention_heads=num_heads,
            hidden_size=head_dim,
            n_head=num_heads,
        )
        self.dtype = torch.bfloat16

    def __call__(self, *, input_ids, attention_mask, past_key_values, position_ids, use_cache: bool):
        # Batch size
        B = int(input_ids.shape[0])
        # Tm is the total mask length (prefill: max sequence len; decode: prev_max + 1)
        Tm = int(attention_mask.shape[1])

        # per-sample visible token count for this call
        t_new = attention_mask.sum(dim=1).cpu().tolist()

        # base_seed per request: sum of visible input_ids (works for prefill & decode)
        # If attention_mask is longer than input_ids (prefill), align by taking the last columns
        if attention_mask.shape[1] >= input_ids.shape[1]:
            visible_mask = attention_mask[:, -input_ids.shape[1]:]
        else:
            visible_mask = attention_mask  # shouldn't happen, but keep safe
        visible = (input_ids * visible_mask).long()
        base = visible.sum(dim=1).cpu().tolist()

        # logits [B, 1, V]
        logits = torch.zeros((B, 1, self.vocab_size), dtype=torch.float32)
        for b in range(B):
            peak = int(base[b]) % self.vocab_size
            logits[b, 0, peak] = 1.0

        # Build a padded deterministic past for each layer with shape [B, H, Tm, D]
        past: List[Tuple[torch.Tensor, torch.Tensor]] = []
        for l in range(self.L):
            K_batch = torch.zeros((B, self.H, Tm, self.D), dtype=torch.float32)
            V_batch = torch.zeros_like(K_batch)
            for b in range(B):
                t = int(t_new[b])
                if t == 0:
                    continue
                base_b = float(base[b] + l * 100)
                # Construct the "real" tail slice [1, H, t, D]
                K_tail = base_b + (
                    torch.arange(self.H).view(1, self.H, 1, 1)
                    + torch.arange(t).view(1, 1, t, 1) / 10.0
                    + torch.arange(self.D).view(1, 1, 1, self.D) / 1000.0
                )
                V_tail = (base_b * 2) + (
                    torch.arange(self.H).view(1, self.H, 1, 1)
                    + torch.arange(t).view(1, 1, t, 1) / 20.0
                    + torch.arange(self.D).view(1, 1, 1, self.D) / 2000.0
                )
                # Place it at the end (left padding behaviour)
                K_batch[b : b + 1, :, -t:, :] = K_tail
                V_batch[b : b + 1, :, -t:, :] = V_tail
            past.append((K_batch.contiguous(), V_batch.contiguous()))

        self.calls.append({
            "input_ids": input_ids.detach().clone(),
            "attention_mask": attention_mask.detach().clone(),
            "position_ids": position_ids.detach().clone(),
            "past_key_values": past_key_values,
        })
        return SimpleNamespace(logits=logits, past_key_values=tuple(past))


@pytest.fixture()
def tiny_tok():
    return TinyTok()


@pytest.fixture()
def fake_bundle():
    return FakeBundle(device=torch.device("cpu"), model=FakeModel())


@pytest.fixture(autouse=True)
def patch_registry_and_metrics(fake_bundle):
    """Patch REGISTRY.load and metrics so tests don't hit real deps."""
    with (
        patch("myserve.scheduler.REGISTRY") as mock_registry,
        patch("myserve.scheduler.REQ_TOTAL") as mock_req_total,
        patch("myserve.scheduler.TOKENS_TOTAL") as mock_tok_total,
        patch("myserve.scheduler.TTFT_HIST") as mock_ttft,
    ):
        mock_registry.load.return_value = fake_bundle
        yield {
            "REGISTRY": mock_registry,
            "REQ_TOTAL": mock_req_total,
            "TOKENS_TOTAL": mock_tok_total,
            "TTFT_HIST": mock_ttft,
        }


@pytest.fixture()
def fixed_sampler():
    """Return a sampler that always picks a fixed token id."""
    # Always return token id 17 and placeholder logprobs
    with patch("myserve.scheduler.sample_next") as mock:
        next_id = torch.tensor([[17]], dtype=torch.long)
        mock.return_value = (
            next_id,
            torch.tensor([[0.0]]),  # chosen_lp (unused)
            torch.tensor([[0.0]]),  # logprobs (unused)
        )
        yield mock


# ---------------------------
# Dataclass & small helpers
# ---------------------------

class TestGenRequestBasics:
    def test_dataclass_defaults_and_fields(self, tiny_tok):
        outq = SimpleNamespace(put=AsyncMock())
        req = GenRequest(
            model_name="fake-model",
            prompt="Hello world",
            tok=tiny_tok,
            max_new=3,
            cfg=SamplerCfg(),
            eos_ids=[2],
            seed=123,
            outq=outq,
        )
        assert req.model_name == "fake-model"
        assert req.prompt == "Hello world"
        assert req.max_new == 3
        assert req.eos_ids == [2]
        assert req.seed == 123
        assert req.done is False
        assert req.kv is None and req.input_ids is None and req.generated is None
        assert req.first_token_s is None
        assert req.id.startswith("req_")
        assert isinstance(req.created_s, float)


# ---------------------------
# Unit tests of handle_* functions
# ---------------------------

def _handle_prefill_req_and_add_to_batch(prefill_batch: List[GenRequest], r: GenRequest) -> None:
    _handle_prefill_req(r)
    prefill_batch.append(r)

class TestHandleFunctions:
    def make_req(self, tiny_tok, prompt: str, outq=None, eos_ids=None, max_new=3):
        return GenRequest(
            model_name="fake-model",
            prompt=prompt,
            tok=tiny_tok,
            max_new=max_new,
            cfg=SamplerCfg(),
            eos_ids=eos_ids,
            seed=42,
            outq=SimpleNamespace(put=AsyncMock()) if outq is None else outq,
        )

    def test_handle_prefill_req_sets_state(self, tiny_tok):
        r = self.make_req(tiny_tok, "Hello")
        batch: List[GenRequest] = []
        _handle_prefill_req_and_add_to_batch(batch, r)
        assert batch == [r]
        assert r.input_ids is not None
        torch.testing.assert_close(r.generated, r.input_ids)
        assert r.kv is None

    def test_batched_vs_serial_prefill_equivalence(self, tiny_tok, fixed_sampler, fake_bundle):
        # Two requests of different lengths
        r_a = self.make_req(tiny_tok, "Hello", eos_ids=[9999])
        r_b = self.make_req(tiny_tok, "Test 1 2 3", eos_ids=[9999])

        # PREFILL: batched
        batch12: List[GenRequest] = []
        _handle_prefill_req_and_add_to_batch(batch12, r_a)
        _handle_prefill_req_and_add_to_batch(batch12, r_b)
        logits12, past12, lengths12 = _handle_prefill_batch(batch12)

        # PREFILL: serial for each prompt
        b1: List[GenRequest] = []
        r1 = self.make_req(tiny_tok, "Hello", eos_ids=[9999])
        _handle_prefill_req_and_add_to_batch(b1, r1)
        logits1, past1, lengths1 = _handle_prefill_batch(b1)

        b2: List[GenRequest] = []
        r2 = self.make_req(tiny_tok, "Test 1 2 3", eos_ids=[9999])
        _handle_prefill_req_and_add_to_batch(b2, r2)
        logits2, past2, lengths2 = _handle_prefill_batch(b2)

        # Lengths match expectations: [1, 4]
        torch.testing.assert_close(lengths12.cpu(), torch.tensor([1, 4], dtype=torch.long))
        torch.testing.assert_close(lengths1.cpu(), torch.tensor([1], dtype=torch.long))
        torch.testing.assert_close(lengths2.cpu(), torch.tensor([4], dtype=torch.long))

        # Logits (last-token) of batched vs serial must match
        V = tiny_tok.vocab_size
        assert logits12.shape == (2, V)
        assert logits1.shape == (1, V)
        assert logits2.shape == (1, V)
        torch.testing.assert_close(logits12[0, :], logits1[0, :])
        torch.testing.assert_close(logits12[1, :], logits2[0, :])

        # Split per-request past from the batched past and compare with serial past
        def split(past, idx, t):
            return tuple((K[idx:idx+1, :, -t:, :], V[idx:idx+1, :, -t:, :]) for (K, V) in past)

        split12_1 = split(past12, 0, int(lengths12[0]))
        split12_2 = split(past12, 1, int(lengths12[1]))
        split1 = split(past1, 0, int(lengths1[0]))
        split2 = split(past2, 0, int(lengths2[0]))

        # Sample one token to populate r.kv and r.generated
        nid12_1, piece12_1 = _handle_out(r_a, logits12[0:1, :], split12_1)
        nid12_2, piece12_2 = _handle_out(r_b, logits12[1:2, :], split12_2)
        nid1, piece1 = _handle_out(r1, logits1[0:1, :], split1)
        nid2, piece2 = _handle_out(r2, logits2[0:1, :], split2)

        torch.testing.assert_close(nid12_1, nid1)
        torch.testing.assert_close(nid12_2, nid2)
        assert piece12_1 == piece1
        assert piece12_2 == piece2

        # The cached K/V written into the requests must match across batched vs serial
        L = fake_bundle.model.L
        for l in range(L):
            torch.testing.assert_close(r_a.kv.layers[l][0], r1.kv.layers[l][0])
            torch.testing.assert_close(r_a.kv.layers[l][1], r1.kv.layers[l][1])
            torch.testing.assert_close(r_b.kv.layers[l][0], r2.kv.layers[l][0])
            torch.testing.assert_close(r_b.kv.layers[l][1], r2.kv.layers[l][1])

    def test_batched_vs_serial_decode_logits_and_lengths(self, tiny_tok, fixed_sampler):
        # Prepare prefilled requests
        r_a = self.make_req(tiny_tok, "Hello", eos_ids=[9999])
        r_b = self.make_req(tiny_tok, "Test 1 2 3", eos_ids=[9999])
        batch: List[GenRequest] = []
        _handle_prefill_req_and_add_to_batch(batch, r_a)
        _handle_prefill_req_and_add_to_batch(batch, r_b)
        logits12, past12, lengths12 = _handle_prefill_batch(batch)

        # Populate kv & generated by sampling one token (prefill step)
        def split(past, idx, t):
            return tuple((K[idx:idx+1, :, -t:, :], V[idx:idx+1, :, -t:, :]) for (K, V) in past)
        _handle_out(r_a, logits12[0:1, :], split(past12, 0, int(lengths12[0])))
        _handle_out(r_b, logits12[1:2, :], split(past12, 1, int(lengths12[1])))

        # Decode batched
        d_logits12, d_past12, d_lengths12 = _handle_decode_batch([r_a, r_b])
        # Decode individually
        d_logits1, d_past1, d_lengths1 = _handle_decode_batch([r_a])
        d_logits2, d_past2, d_lengths2 = _handle_decode_batch([r_b])

        # Shapes
        V = tiny_tok.vocab_size
        assert d_logits12.shape == (2, V)
        assert d_logits1.shape == (1, V)
        assert d_logits2.shape == (1, V)

        # The returned lengths are previous lengths + 1
        torch.testing.assert_close(d_lengths12.cpu(), torch.tensor([2, 5], dtype=torch.long))
        torch.testing.assert_close(d_lengths1.cpu(), torch.tensor([2], dtype=torch.long))
        torch.testing.assert_close(d_lengths2.cpu(), torch.tensor([5], dtype=torch.long))

        # Batched vs serial logits must match per request
        torch.testing.assert_close(d_logits12[0, :], d_logits1[0, :])
        torch.testing.assert_close(d_logits12[1, :], d_logits2[0, :])


# ---------------------------
# Integration tests of Scheduler loop
# ---------------------------

class TestSchedulerLoop:
    def make_req(self, tok, prompt: str, *, max_new: int, eos_ids: Optional[List[int]]):
        outq = SimpleNamespace(put=AsyncMock())
        return GenRequest(
            model_name="fake-model",
            prompt=prompt,
            tok=tok,
            max_new=max_new,
            cfg=SamplerCfg(),
            eos_ids=eos_ids,
            seed=7,
            outq=outq,
        )

    @pytest.mark.asyncio
    async def test_start_stop_idempotent_and_reset(self):
        s = Scheduler(prefill_bs=2, decode_bs=2)
        assert s._task is None
        await s.start()
        assert s._task is not None and not s._task.done()
        # Start twice should be ok and not create a new task
        t0 = s._task
        await s.start()
        assert s._task is t0
        await s.stop()
        assert s._task is None
        assert isinstance(s._ingress, asyncio.Queue) and s._ingress.empty()
        assert isinstance(s._active, deque) and len(s._active) == 0

    @pytest.mark.asyncio
    async def test_end_to_end_generation_eos_and_max_new(self, tiny_tok, fixed_sampler, patch_registry_and_metrics):
        s = Scheduler(prefill_bs=2, decode_bs=2)

        # One request will stop via EOS immediately; another will stop via max_new
        eos_token = 17  # matches fixed_sampler
        r_eos = self.make_req(tiny_tok, "Hello", max_new=10, eos_ids=[eos_token])
        r_mn = self.make_req(tiny_tok, "Test 1 2 3", max_new=2, eos_ids=None)

        await s.submit(r_eos)
        await s.submit(r_mn)
        await s.start()

        # Let the loop run a bit to process the requests
        for _ in range(200):
            await asyncio.sleep(0.0)
            eos_calls = r_eos.outq.put.await_args_list
            mn_calls = r_mn.outq.put.await_args_list
            if any(arg[0][0] is None for arg in eos_calls) and any(arg[0][0] is None for arg in mn_calls):
                break

        # Stop the scheduler cleanly for assertions
        await s.stop()

        # Helper to collect pieces (string tokens) until sentinel None
        def pieces_from_calls(calls):
            out = []
            for c in calls:
                (x,), _ = c
                out.append(x)
            return out

        eos_pieces = pieces_from_calls(r_eos.outq.put.await_args_list)
        mn_pieces = pieces_from_calls(r_mn.outq.put.await_args_list)

        # EOS request: exactly one generated token then None
        assert eos_pieces[-1] is None
        assert all(isinstance(p, str) for p in eos_pieces[:-1])
        assert len(eos_pieces[:-1]) == 1

        # Max-new=2: exactly two generated tokens then None
        assert mn_pieces[-1] is None
        assert all(isinstance(p, str) for p in mn_pieces[:-1])
        assert len(mn_pieces[:-1]) == 2

        # Metrics were touched (labels called & incremented)
        labels = patch_registry_and_metrics["REQ_TOTAL"].labels
        assert [c.kwargs for c in labels.call_args_list] == [
            {"model": "fake-model"},
            {"model": "fake-model"},
        ]
        assert patch_registry_and_metrics["TOKENS_TOTAL"].labels.return_value.inc.call_count >= 3
        # TTFT observed at least once per request
        assert patch_registry_and_metrics["TTFT_HIST"].labels.return_value.observe.call_count >= 2

    @pytest.mark.asyncio
    async def test_decode_uses_last_generated_token_and_padded_past(self, tiny_tok, fixed_sampler, fake_bundle):
        s = Scheduler(prefill_bs=2, decode_bs=2)
        r1 = self.make_req(tiny_tok, "Hello", max_new=2, eos_ids=None)
        r2 = self.make_req(tiny_tok, "Test 1 2 3", max_new=2, eos_ids=None)
        await s.submit(r1)
        await s.submit(r2)

        await s.start()
        # Allow a couple of loop iterations
        for _ in range(50):
            await asyncio.sleep(0.0)
            # once a couple of puts are recorded, we're good
            if r1.outq.put.await_count >= 1 and r2.outq.put.await_count >= 1:
                break
        # Stop to make assertions about FakeModel recorded calls
        await s.stop()

        # We expect at least a prefill call and one decode call
        calls = fake_bundle.model.calls
        assert len(calls) >= 2
        # The last decode call should have input_ids of shape [B,1]
        last = calls[-1]
        iid = last["input_ids"]
        assert iid.ndim == 2 and iid.shape[1] == 1
        # attention_mask should include the +1 slot (lengths+1)
        am = last["attention_mask"]
        assert am.shape[1] >= 2


# ---------------------------
# Submit/ingress tests
# ---------------------------

class TestSubmitAndIngress:
    @pytest.mark.asyncio
    async def test_submit_enqueues_and_metrics(self, tiny_tok, patch_registry_and_metrics):
        s = Scheduler()
        outq = SimpleNamespace(put=AsyncMock())
        req = GenRequest(
            model_name="fake-model",
            prompt="Hello",
            tok=tiny_tok,
            max_new=1,
            cfg=SamplerCfg(),
            eos_ids=None,
            seed=None,
            outq=outq,
        )
        await s.submit(req)
        assert not s._ingress.empty()
        got = await s._ingress.get()
        assert got is req
        patch_registry_and_metrics["REQ_TOTAL"].labels.assert_called_once_with(model="fake-model")
        patch_registry_and_metrics["REQ_TOTAL"].labels.return_value.inc.assert_called_once()
