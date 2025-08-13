import asyncio
import json
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
from myserve.core.generate import greedy_generate

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
    prompt = render_messages(req.messages)

    created = int(time.time())
    rid = f"chatcmpl_{uuid.uuid4().hex[:24]}"
    model_name = req.model

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
        tokenizer = get_tokenizer(model_name)
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

    # --- Real model backend (this post) ---
    tok = bundle.tokenizer
    eos = tok.eos_token_id

    enc = tok(prompt, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"].to(bundle.device)

    max_new = max(1, int(req.max_tokens or 16))

    if req.stream:
        async def model_stream() -> AsyncGenerator[bytes, None]:
            yield _sse_chunk(rid, model_name, role="assistant")
            # We decode token-by-token to stream pieces.
            generated = input_ids
            for i in range(max_new):
                with torch.no_grad():
                    logits = bundle.model(generated).logits
                    next_id = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                generated = torch.cat([generated, next_id], dim=1)
                piece = tok.decode(next_id[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                if piece:
                    yield _sse_chunk(rid, model_name, content=piece)
                if eos is not None and int(next_id.item()) == eos:
                    break
                await asyncio.sleep(0.0)
            yield _sse_done(rid, model_name)
        return StreamingResponse(model_stream(), media_type="text/event-stream")

    # Non‑stream: run the simple greedy helper for clarity
    out = greedy_generate(bundle.model, input_ids, max_new_tokens=max_new, eos_token_id=eos)
    new_tokens = out[0, input_ids.size(1):]
    text = tok.decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return JSONResponse(_non_stream_payload(rid, model_name, text))

# helpers ------------------------------------------------------------

def _sse_chunk(rid: str, model: str, content: str | None = None, role: str | None = None) -> bytes:
    delta = {}
    if role is not None:
        delta["role"] = role
    if content:
        delta["content"] = content
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