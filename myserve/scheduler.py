from __future__ import annotations
import asyncio, time, uuid
from dataclasses import dataclass, field
from typing import List, Optional, Deque, Tuple, Union
from collections import deque
import torch
from transformers import DynamicCache

from .core.models import REGISTRY
from .core.kv import KVCache
from .core.sampling import SamplerCfg, sample_next
from .core.collate import pad_past, split_past, pad_sequences, Past
from .core.memory import MemManager, _free_bytes
from .metrics import REQ_TOTAL, TOKENS_TOTAL, TTFT_HIST, MEM_RESERVED_BYTES, MEM_FREE_BYTES

@dataclass
class GenRequest:
    model_name: str
    prompt: str
    tok: any
    max_new: int
    cfg: SamplerCfg
    eos_ids: Optional[List[int]]
    seed: Optional[int]
    # runtime
    outq: asyncio.Queue
    id: str = field(default_factory=lambda: f"req_{uuid.uuid4().hex[:8]}")
    created_s: float = field(default_factory=time.time)
    # state
    input_ids: Optional[torch.Tensor] = None
    generated: Optional[torch.Tensor] = None
    kv: Optional[KVCache] = None
    done: bool = False
    first_token_s: Optional[float] = None
    want_logprobs: bool = False
    top_logprobs: int = 0

def _handle_prefill_req(prefill_batch: List[GenRequest], r: GenRequest) -> None:
    bundle = REGISTRY.load(r.model_name)
    enc = r.tok(r.prompt, return_tensors="pt", add_special_tokens=False)
    r.input_ids = enc["input_ids"].to(bundle.device)
    r.generated = r.input_ids
    r.kv = None
    prefill_batch.append(r)

@torch.inference_mode()
def _handle_batch_common(model, input_ids, past_key_values, attention_mask, position_ids, lengths):
    out = model(
        input_ids=input_ids,
        past_key_values=past_key_values,
        attention_mask=attention_mask,
        position_ids=position_ids,
        use_cache=True,
    )
    logits = out.logits[:, -1, :]
    new_past = out.past_key_values
    return logits, new_past, lengths

def _handle_prefill_batch(prefill_batch: List[GenRequest]) -> Tuple[torch.Tensor, Past, torch.Tensor]:
    bundle = REGISTRY.load(prefill_batch[0].model_name)

    input_ids, attention_mask, position_ids, lengths = pad_sequences([r.input_ids.squeeze(0) for r in prefill_batch], prefill_batch[0].tok.pad_token_id)
    past_key_values = None

    return _handle_batch_common(bundle.model, input_ids, past_key_values, attention_mask, position_ids, lengths)

def _handle_decode_batch(batch: List[GenRequest]) -> Tuple[torch.Tensor, Past, torch.Tensor]:
    bundle = REGISTRY.load(batch[0].model_name)

    input_ids = torch.cat([r.generated[:, -1:] for r in batch], dim=0)  # [B,1]
    padded_past, lengths, attention_mask, position_ids = pad_past([kv.layers for kv in [r.kv for r in batch]])
    lengths = lengths + 1
    past_key_values = DynamicCache.from_legacy_cache(padded_past)

    return _handle_batch_common(bundle.model, input_ids, past_key_values, attention_mask, position_ids, lengths)

def _handle_out(r: GenRequest, logits: torch.Tensor, past) -> Tuple[torch.Tensor, Union[str, dict]]:
    bundle = REGISTRY.load(r.model_name)
    next_ids, chosen_lp, logprobs = sample_next(
        logits, r.cfg, r.generated, _make_gen(bundle.device, r.seed)
    )

    # update request state
    r.generated = torch.cat([r.generated, next_ids.view(1, 1)], dim=1)
    r.kv = KVCache.from_past(past)

    # text piece
    piece = r.tok.decode(next_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    # default payload is just the text
    payload: Union[str, dict] = piece

    # optional logprobs
    if r.want_logprobs:
        k = max(0, int(r.top_logprobs or 0))
        tops = []
        if k > 0:
            topv, topi = torch.topk(logprobs[0], k)
            tops = [
                {
                    "token": r.tok.decode([int(i)], skip_special_tokens=True, clean_up_tokenization_spaces=False),
                    "logprob": float(v),
                }
                for v, i in zip(topv.tolist(), topi.tolist())
            ]
        payload = {
            "piece": piece,
            "logprobs": {
                "content": [{
                    "token": piece,
                    "bytes": list(piece.encode("utf-8", errors="ignore")),
                    "logprob": float(chosen_lp[0]),
                    "top_logprobs": tops,
                }]
            },
        }

    TOKENS_TOTAL.labels(model=r.model_name).inc()
    return next_ids, payload

class Scheduler:
    def __init__(self, device: str = "auto", prefill_bs: int = 4, decode_bs: int = 8):
        self.device = device
        self.prefill_bs = prefill_bs
        self.decode_bs = decode_bs
        self._ingress: asyncio.Queue[GenRequest] = asyncio.Queue()
        self._active: Deque[GenRequest] = deque()
        self._task: Optional[asyncio.Task] = None
        self._mem: dict[str, MemManager] = {}
        self._waiting: Deque[GenRequest] = deque()

    def _get_mem(self, model_name: str) -> MemManager:
        if model_name in self._mem:
            return self._mem[model_name]
        bundle = REGISTRY.load(model_name)  # cfg/dtype/device from the actual model
        mm = MemManager(model_name=model_name, device=bundle.device, cfg=bundle.model.config, dtype=bundle.model.dtype)
        self._mem[model_name] = mm
        return mm

    def _try_admit_prefill(self, r: GenRequest, prefill_batch: List[GenRequest]) -> bool:
        bundle = REGISTRY.load(r.model_name)
        enc = r.tok(r.prompt, return_tensors="pt", add_special_tokens=False)
        r.input_ids = enc["input_ids"].to(bundle.device)
        r.generated = r.input_ids
        r.kv = None

        mm = self._get_mem(r.model_name)
        prompt_len = int(r.input_ids.size(1))
        if not mm.reserve(r.id, prompt_len, r.max_new):
            # not enough memory → leave it for later
            return False

        MEM_RESERVED_BYTES.labels(model=r.model_name).set(mm.reserved_bytes())
        MEM_FREE_BYTES.set(_free_bytes(mm.device))
        prefill_batch.append(r)
        self._active.append(r)
        return True

    async def start(self):
        if self._task is None:
            self._task = asyncio.create_task(self._loop())
            # surface background exceptions in pytest instead of hanging
            self._task.add_done_callback(lambda t: t.result())

    async def stop(self):
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            # <-- IMPORTANT: allow a clean restart in the next test
            self._task = None
            # optional: reset transient state so each test starts fresh
            self._ingress = asyncio.Queue()
            self._active.clear()

    async def submit(self, req: GenRequest):
        REQ_TOTAL.labels(model=req.model_name).inc()
        await self._ingress.put(req)

    @torch.inference_mode()
    async def _loop(self):
        tick = 0
        while True:
            tick += 1
            # 1) admit NEW → PREFILL batch
            prefill_batch: List[GenRequest] = []

            # first, try from the waiting list
            while len(prefill_batch) < self.prefill_bs and self._waiting:
                r = self._waiting[0]
                if self._try_admit_prefill(r, prefill_batch):
                    self._waiting.popleft()
                else:
                    break  # still no memory; keep waiting

            while len(prefill_batch) < self.prefill_bs and not self._ingress.empty():
                r = await self._ingress.get()
                if not self._try_admit_prefill(r, prefill_batch):
                    self._waiting.append(r)  # park it until memory frees

            if prefill_batch:
                logits, past, lengths = _handle_prefill_batch(prefill_batch)

                # sample one token for each request
                for i, r in enumerate(prefill_batch):
                    split = split_past(tuple((K[i:i+1], V[i:i+1]) for (K, V) in past), lengths[i:i+1])
                    next_ids, payload = _handle_out(
                        r, 
                        logits[i:i+1, :], 
                        split[0], 
                    )
                    if r.first_token_s is None:
                        r.first_token_s = time.time()
                        TTFT_HIST.labels(model=r.model_name).observe(r.first_token_s - r.created_s)
                    await r.outq.put(payload)
                    if r.eos_ids and int(next_ids[0]) in r.eos_ids:
                        r.done = True

            # 2) decode round robin
            decode_candidates = [r for r in list(self._active) if (not r.done and r.kv is not None and r.generated is not None)]
            if decode_candidates:
                # rotate for fairness
                self._active.rotate(1)
                decode_batch = decode_candidates[: self.decode_bs]
                logits, past, lengths = _handle_decode_batch(decode_batch)

                for i, r in enumerate(decode_batch):
                    split = split_past(tuple((K[i:i+1], V[i:i+1]) for (K, V) in past), lengths[i:i+1])
                    next_ids, payload = _handle_out(
                        r, 
                        logits[i:i+1, :], 
                        split[0],
                    )
                    await r.outq.put(payload)
                    if r.generated.size(1) - r.input_ids.size(1) >= r.max_new:
                        r.done = True
                    if r.eos_ids and int(next_ids[0]) in r.eos_ids:
                        r.done = True

            # 3) retire completed requests
            while self._active and self._active[0].done:
                r = self._active.popleft()
                # release reservation
                mm = self._get_mem(r.model_name)
                mm.release(r.id)
                MEM_RESERVED_BYTES.labels(model=r.model_name).set(mm.reserved_bytes())
                MEM_FREE_BYTES.set(_free_bytes(mm.device))
                await r.outq.put(None)

            # # 4) tiny sleep to yield
            await asyncio.sleep(0.0)


def _make_gen(device, seed):
    if seed is None:
        return None
    g = torch.Generator(device=device)
    g.manual_seed(int(seed))
    return g
