import asyncio
import json
import math
import time
import uuid
import os
from typing import AsyncGenerator

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from myserve.api.openai_types import ChatCompletionRequest
from myserve.core.tokenizer import get_tokenizer, render_messages
from myserve.core.models import REGISTRY
from myserve.core.generate import sample_generate
from myserve.core.sampling import SamplerCfg, sample_next


app = FastAPI(title="myserve")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/healthz")
def healthz():
    return {"status": "ok"}

USE_ECHO_FALLBACK = os.getenv("MYSERVE_FORCE_ECHO", "0") == "1"
DEFAULT_DTYPE = os.getenv("MYSERVE_DTYPE", "auto")
DEFAULT_DEVICE = os.getenv("MYSERVE_DEVICE", "auto")

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    model_name = req.model
    tokenizer = get_tokenizer(model_name)
    prompt = render_messages(tokenizer, req.messages)

    created = int(time.time())
    rid = f"chatcmpl_{uuid.uuid4().hex[:24]}"

    # Try to load a real model unless echo is forced
    bundle = None
    if not USE_ECHO_FALLBACK:
        try:
            bundle = REGISTRY.load(model_name, dtype=DEFAULT_DTYPE, device=DEFAULT_DEVICE)
        except Exception as e:
            # fallback silently to echo mode; in real servers you would surface a 400
            bundle = None

    if bundle is None:
        # --- Echo backend (Post 1) ---
        input_ids = tokenizer.encode(prompt, add_special_tokens=False)
        max_new = max(0, int(req.max_tokens or 0)) or 128
        output_ids = input_ids[:max_new]

        if req.stream:
            async def echo_stream() -> AsyncGenerator[bytes, None]:
                yield _sse_chunk(rid, model_name, role="assistant")
                for tid in output_ids:
                    piece = tokenizer.decode([tid], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                    if piece:
                        yield _sse_chunk(rid, model_name, content=piece)
                    await asyncio.sleep(0.0)
                yield _sse_done(rid, model_name)
            return StreamingResponse(echo_stream(), media_type="text/event-stream")

        text = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        return JSONResponse(_non_stream_payload(rid, model_name, text))

    # real model path (after bundle is loaded)
    tok = bundle.tokenizer
    eos = tok.eos_token_id
    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(bundle.device)

    # sampler config
    cfg = SamplerCfg(
        temperature=float(req.temperature or 1.0),
        top_p=float(req.top_p or 1.0),
        top_k=int(req.top_k) if req.top_k else None,
        presence_penalty=float(req.presence_penalty or 0.0),
        frequency_penalty=float(req.frequency_penalty or 0.0),
        top_logprobs=int(req.top_logprobs or 0),
    )
    max_new = max(1, int(req.max_tokens or 16))

    # seeded generator (device‑specific to avoid CPU/CUDA mismatch)
    gen = None
    if req.seed is not None:
        gen = torch.Generator(device=bundle.device.type)
        gen.manual_seed(int(req.seed))

    if req.stream:
        async def stream() -> AsyncGenerator[bytes, None]:
            yield _sse_chunk(rid, model_name, role="assistant")
            generated = input_ids
            for _ in range(max_new):
                logits = bundle.model(generated).logits[:, -1, :]
                next_ids, chosen_lp, logprobs = sample_next(logits, cfg, generated, gen)
                generated = torch.cat([generated, next_ids.unsqueeze(1)], dim=1)
                piece = tok.decode(next_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                extra = None
                if req.logprobs:
                    k = int(req.top_logprobs or 0)
                    content = [{
                        "token": piece,
                        "bytes": list(piece.encode("utf-8", errors="ignore")),
                        "logprob": float(chosen_lp[0]),
                        "top_logprobs": (
                            [{"token": tok.decode([int(i)], skip_special_tokens=True, clean_up_tokenization_spaces=False),
                               "logprob": float(v)} for v, i in zip(*torch.topk(logprobs[0], k))]
                            if k > 0 else []
                        ),
                    }]
                    extra = {"logprobs": {"content": content}}
                yield _sse_chunk(rid, model_name, content=piece, extra=extra)
                if eos is not None and int(next_ids[0]) == eos:
                    break
                await asyncio.sleep(0.0)
            yield _sse_done(rid, model_name)
        return StreamingResponse(stream(), media_type="text/event-stream")

    # non‑stream
    all_ids, per_step = sample_generate(
        bundle.model, input_ids, max_new_tokens=max_new, eos_token_id=eos,
        cfg=cfg, gen=gen, collect_logprobs=bool(req.logprobs)
    )
    new_ids = all_ids[0, input_ids.size(1):]
    text = tok.decode(new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    payload = _non_stream_payload(rid, model_name, text)
    if req.logprobs:
        tokens = []
        for step in per_step:
            item = step[0]
            tstr = tok.decode([item["id"]], skip_special_tokens=True, clean_up_tokenization_spaces=False)
            toks = {"token": tstr, "bytes": list(tstr.encode("utf-8", errors="ignore")), "logprob": item["logprob"]}
            if cfg.top_logprobs:
                toks["top_logprobs"] = [
                    {"token": tok.decode([tid], skip_special_tokens=True, clean_up_tokenization_spaces=False), "logprob": (lp if str(math.fabs(lp)) != "inf" else str(lp))}
                    for (tid, lp) in item.get("top_logprobs", [])
                ]
            tokens.append(toks)
        payload["logprobs"] = {"content": tokens}
    return JSONResponse(payload)


# helpers ------------------------------------------------------------

def _sse_chunk(rid: str, model: str, content: str | None = None, role: str | None = None, extra: dict | None = None) -> bytes:
    delta = {}
    if role is not None:
        delta["role"] = role
    if content is not None:
        delta["content"] = content
    if extra:
        delta.update(extra)
    obj = {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": None,
        }],
    }
    return f"data: {json.dumps(obj, ensure_ascii=False)}\n\n".encode()


def _sse_done(rid: str, model: str) -> bytes:
    obj = {
        "id": rid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
    }
    return (f"data: {json.dumps(obj, ensure_ascii=False)}\n\n" + "data: [DONE]\n\n").encode()


def _non_stream_payload(rid: str, model: str, text: str) -> dict:
    return {
        "id": rid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
    }
