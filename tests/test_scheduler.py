import pytest
import asyncio
import torch
from unittest.mock import Mock, AsyncMock, patch, call
from collections import deque
from types import SimpleNamespace
from typing import List

from myserve.scheduler import GenRequest, Scheduler, _handle_prefill_req, _handle_prefill_batch, _handle_out, _handle_decode_batch
from myserve.core.sampling import SamplerCfg
from myserve.core.collate import split_past
from transformers import AutoTokenizer, AutoConfig


class TestGenRequest:
    """Test the GenRequest dataclass"""
    
    def test_gen_request_creation(self):
        """Test basic GenRequest creation with required fields"""
        mock_tokenizer = Mock()
        mock_queue = AsyncMock()
        mock_cfg = SamplerCfg()
        
        req = GenRequest(
            model_name="test-model",
            prompt="Hello world",
            tok=mock_tokenizer,
            max_new=10,
            cfg=mock_cfg,
            eos_ids=[2],
            seed=42,
            outq=mock_queue
        )
        
        assert req.model_name == "test-model"
        assert req.prompt == "Hello world"
        assert req.tok == mock_tokenizer
        assert req.max_new == 10
        assert req.cfg == mock_cfg
        assert req.eos_ids == [2]
        assert req.seed == 42
        assert req.outq == mock_queue
        assert req.done is False
        assert req.first_token_s is None
        assert req.input_ids is None
        assert req.generated is None
        assert req.kv is None
        assert req.id.startswith("req_")
        assert isinstance(req.created_s, float)
    
    def test_gen_request_defaults(self):
        """Test GenRequest creation with default values"""
        mock_tokenizer = Mock()
        mock_queue = AsyncMock()
        mock_cfg = SamplerCfg()
        
        req = GenRequest(
            model_name="test-model",
            prompt="Test",
            tok=mock_tokenizer,
            max_new=5,
            cfg=mock_cfg,
            eos_ids=None,
            seed=None,
            outq=mock_queue
        )
        
        assert req.eos_ids is None
        assert req.seed is None
        assert req.done is False
        assert req.first_token_s is None
        assert req.input_ids is None
        assert req.generated is None
        assert req.kv is None


class TestScheduler:
    """Test the Scheduler class"""
    
    def test_scheduler_initialization(self):
        """Test Scheduler initialization with default and custom parameters"""
        # Test default initialization
        scheduler = Scheduler()
        assert scheduler.device == "auto"
        assert scheduler.prefill_bs == 4
        assert scheduler.decode_bs == 8
        assert isinstance(scheduler._ingress, asyncio.Queue)
        assert isinstance(scheduler._active, deque)
        assert scheduler._task is None
        
        # Test custom initialization
        scheduler = Scheduler(device="cpu", prefill_bs=8, decode_bs=16)
        assert scheduler.device == "cpu"
        assert scheduler.prefill_bs == 8
        assert scheduler.decode_bs == 16
    
    @pytest.mark.asyncio
    async def test_scheduler_start(self):
        """Test scheduler start method"""
        scheduler = Scheduler()
        assert scheduler._task is None
        
        await scheduler.start()
        assert scheduler._task is not None
        assert not scheduler._task.done()
        
        # Test that calling start again doesn't create multiple tasks
        original_task = scheduler._task
        await scheduler.start()
        assert scheduler._task is original_task
        
        # Cleanup
        scheduler._task.cancel()
        try:
            await scheduler._task
        except asyncio.CancelledError:
            pass
    
    @pytest.mark.asyncio
    async def test_scheduler_submit(self):
        """Test scheduler submit method"""
        scheduler = Scheduler()
        mock_tokenizer = Mock()
        mock_queue = AsyncMock()
        mock_cfg = SamplerCfg()
        
        req = GenRequest(
            model_name="test-model",
            prompt="Test prompt",
            tok=mock_tokenizer,
            max_new=5,
            cfg=mock_cfg,
            eos_ids=None,
            seed=None,
            outq=mock_queue
        )
        
        # Mock the metrics
        with patch('myserve.scheduler.REQ_TOTAL') as mock_req_total:
            await scheduler.submit(req)
            
            # Check that request was added to ingress queue
            assert not scheduler._ingress.empty()
            submitted_req = await scheduler._ingress.get()
            assert submitted_req == req
            
            # Check that metrics were incremented
            mock_req_total.labels.assert_called_once_with(model="test-model")
            mock_req_total.labels.return_value.inc.assert_called_once()


class TestSchedulerLoop:
    """Test the scheduler main loop functionality"""

    tok = AutoTokenizer.from_pretrained("gpt2")
    tok.pad_token_id = tok.eos_token_id
    
    def sample_request(self, prompt, model_name="test-model"):
        """Create a sample GenRequest for testing"""
        mock_queue = AsyncMock()
        mock_cfg = SamplerCfg()
        
        return GenRequest(
            model_name=model_name,
            tok=self.tok,
            prompt=prompt,
            max_new=3,
            cfg=mock_cfg,
            eos_ids=[self.tok.eos_token_id],
            seed=42,
            outq=mock_queue
        )

    @pytest.mark.asyncio
    async def test_prefill(self):
        scheduler = Scheduler(prefill_bs=2, decode_bs=4)
        
        # Mock REGISTRY.load to return a mock bundle
        mock_bundle = Mock()
        mock_bundle.device = torch.device('cpu')
        
        with patch('myserve.scheduler.REGISTRY') as mock_registry, \
             patch('myserve.scheduler.sample_next') as mock_sample_next:
            mock_registry.load.return_value = mock_bundle

            # Mock model output
            B, T, D, V = 2, 5, 100, self.tok.vocab_size
            mock_logits = torch.randn(B, T, V)
            mock_past = (
                (torch.randn(B, 1, T, D), torch.randn(B, 1, T, D)),
                (torch.randn(B, 1, T, D), torch.randn(B, 1, T, D)),
                (torch.randn(B, 1, T, D), torch.randn(B, 1, T, D)),
            )
            stop_logits = torch.zeros(B, T, V)
            stop_logits[:, -1, self.tok.eos_token_id] = 1

            counter = 0
            max_calls = 3

            def model_fn(input_ids, attention_mask, use_cache=False, past_key_values=None, position_ids=None):
                nonlocal counter
                counter += 1
                if counter > max_calls:
                    return SimpleNamespace(logits=stop_logits, past_key_values=mock_past)
                return SimpleNamespace(logits=mock_logits, past_key_values=mock_past)

            mock_bundle.model.side_effect = model_fn

            r1_input_ids = self.tok("Hello", return_tensors="pt", add_special_tokens=False, padding=True, padding_side="left")["input_ids"]
            r2_input_ids = self.tok("Test 1 2 3", return_tensors="pt", add_special_tokens=False, padding=True, padding_side="left")["input_ids"]
            input = self.tok(["Hello", "Test 1 2 3"], return_tensors="pt", add_special_tokens=False, padding=True, padding_side="left")
            input_ids = input["input_ids"]
            attention_mask = input["attention_mask"]

            next_id = self.tok.encode("Hi", return_tensors="pt")
            assert next_id.shape == (1, 1)
            chosen_lp = torch.tensor([[0.1, 0.2, 0.3]])
            logprobs = torch.tensor([[0.1, 0.2, 0.3]])

            mock_sample_next.return_value = (next_id, chosen_lp, logprobs)

            next_ids = torch.cat([next_id, next_id], dim=0)
            
            r1 = self.sample_request("Hello")
            await scheduler.submit(r1)
            r2 = self.sample_request("Test 1 2 3")
            await scheduler.submit(r2)
            
            await scheduler.start()
            
            # Let the scheduler run briefly to process the requests
            await asyncio.sleep(0.1)

            assert len(mock_bundle.model.mock_calls) == 3

            call0 = mock_bundle.model.mock_calls[0]
            torch.testing.assert_close(call0.kwargs["input_ids"], input_ids)
            torch.testing.assert_close(call0.kwargs["attention_mask"], attention_mask)

            call1 = mock_bundle.model.mock_calls[1]
            torch.testing.assert_close(call1.kwargs["input_ids"], next_ids)
            # torch.testing.assert_close(call1.kwargs["past_key_values"].to_legacy_cache(), mock_past)
            r1_len = int(r1_input_ids.shape[1])  # 1
            r2_len = int(r2_input_ids.shape[1])  # 4

            # Slice mock_past per example down to true lengths
            r1_split = tuple((K[0:1, :, :r1_len, :], V[0:1, :, :r1_len, :]) for (K, V) in mock_past)
            r2_split = tuple((K[1:2, :, :r2_len, :], V[1:2, :, :r2_len, :]) for (K, V) in mock_past)

            call2 = mock_bundle.model.mock_calls[2]
            torch.testing.assert_close(call2.kwargs["input_ids"], next_ids)

            torch.testing.assert_close(r1.generated, torch.cat([r1_input_ids, next_id, next_id, next_id], dim=1))
            torch.testing.assert_close(r2.generated, torch.cat([r2_input_ids, next_id, next_id, next_id], dim=1))
            
            r1.outq.put.assert_has_calls([call("Hi"), call("Hi"), call("Hi")])
            r2.outq.put.assert_has_calls([call("Hi"), call("Hi"), call("Hi")])
            
            # Cleanup
            if scheduler._task:
                scheduler._task.cancel()
                try:
                    await scheduler._task
                except asyncio.CancelledError:
                    pass

    torch.inference_mode()
    def test_serial(self):
        model = "gpt2"
        tok = AutoTokenizer.from_pretrained(model)
        config = AutoConfig.from_pretrained(model)
        
        prefill_batch12: List[GenRequest] = []
        _handle_prefill_req(prefill_batch12, self.sample_request("Hello", model))
        _handle_prefill_req(prefill_batch12, self.sample_request("Test 1 2 3", model))
        logits12, past12, lengths12 = _handle_prefill_batch(prefill_batch12)

        prefill_batch1: List[GenRequest] = []
        _handle_prefill_req(prefill_batch1, self.sample_request("Hello", model))
        logits1, past1, lengths1 = _handle_prefill_batch(prefill_batch1)
        
        prefill_batch2: List[GenRequest] = []
        _handle_prefill_req(prefill_batch2, self.sample_request("Test 1 2 3", model))
        logits2, past2, lengths2 = _handle_prefill_batch(prefill_batch2)

        torch.testing.assert_close(lengths12.cpu(), torch.tensor([1, 4], dtype=torch.long))
        torch.testing.assert_close(lengths1.cpu(), torch.tensor([1], dtype=torch.long))
        torch.testing.assert_close(lengths2.cpu(), torch.tensor([4], dtype=torch.long))

        assert logits12.shape == (2, tok.vocab_size), f"logits12.shape: {logits12.shape}"
        assert logits1.shape == (1, tok.vocab_size), f"logits1.shape: {logits1.shape}"
        assert logits2.shape == (1, tok.vocab_size), f"logits2.shape: {logits2.shape}"

        torch.testing.assert_close(logits12[0, :], logits1[0, :], atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(logits12[1, :], logits2[0, :], atol=1e-3, rtol=1e-3)

        assert len(past12) == config.num_hidden_layers, f"len(past12): {len(past12)}"
        assert len(past1) == config.num_hidden_layers, f"len(past1): {len(past1)}"
        assert len(past2) == config.num_hidden_layers, f"len(past2): {len(past2)}"

        split12_1 = split_past(tuple((K[0:1], V[0:1]) for (K, V) in past12), lengths12[0:1])
        split12_2 = split_past(tuple((K[1:2], V[1:2]) for (K, V) in past12), lengths12[1:2])

        split1 = split_past(tuple((K[0:1], V[0:1]) for (K, V) in past1), lengths1[0:1])
        split2 = split_past(tuple((K[0:1], V[0:1]) for (K, V) in past2), lengths2[0:1])

        next_ids12_1, piece12_1 = _handle_out(prefill_batch12[0], logits12[0:1, :], split12_1[0])
        next_ids12_2, piece12_2 = _handle_out(prefill_batch12[1], logits12[1:2, :], split12_2[0])

        next_ids1_1, piece1_1 = _handle_out(prefill_batch1[0], logits1[0:1, :], split1[0])
        next_ids2_1, piece2_1 = _handle_out(prefill_batch2[0], logits2[0:1, :], split2[0])

        torch.testing.assert_close(next_ids12_1, next_ids1_1)
        torch.testing.assert_close(next_ids12_2, next_ids2_1)
        assert piece12_1 == piece1_1
        assert piece12_2 == piece2_1

        for l in range(config.num_hidden_layers):
            torch.testing.assert_close(prefill_batch12[0].kv.layers[l][0], prefill_batch1[0].kv.layers[l][0])
            torch.testing.assert_close(prefill_batch12[0].kv.layers[l][1], prefill_batch1[0].kv.layers[l][1])
            torch.testing.assert_close(prefill_batch12[1].kv.layers[l][0], prefill_batch2[0].kv.layers[l][0])
            torch.testing.assert_close(prefill_batch12[1].kv.layers[l][1], prefill_batch2[0].kv.layers[l][1])

        logits12, new_past12, lengths12 = _handle_decode_batch(prefill_batch12)
        logits1, new_past1, lengths1 = _handle_decode_batch(prefill_batch1)
        logits2, new_past2, lengths2 = _handle_decode_batch(prefill_batch2)

        assert logits12.shape == (2, tok.vocab_size), f"logits12.shape: {logits12.shape}"
        assert logits1.shape == (1, tok.vocab_size), f"logits1.shape: {logits1.shape}"
        assert logits2.shape == (1, tok.vocab_size), f"logits2.shape: {logits2.shape}"

        torch.testing.assert_close(logits12[0, :], logits1[0, :], atol=1e-3, rtol=1e-3)
        torch.testing.assert_close(logits12[1, :], logits2[0, :], atol=1e-3, rtol=1e-3)