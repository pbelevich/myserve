import asyncio
import json
import time
import uuid

import torch
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, PlainTextResponse

from myserve.api.openai_types import ChatCompletionRequest
from myserve.core.tokenizer import render_messages
from myserve.core.models import REGISTRY
from myserve.core.sampling import SamplerCfg
from myserve.scheduler import Scheduler, GenRequest
from myserve.telemetry import setup_tracing
from myserve.metrics import INFLIGHT
from contextlib import asynccontextmanager

from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

tracer = None

SCHED = Scheduler(prefill_bs=16, decode_bs=32)

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    global tracer
    tracer = setup_tracing(app) # no-op if no collector
    await SCHED.start()
    try:
        yield
    finally:
        await SCHED.stop()

app = FastAPI(title="myserve", lifespan=lifespan)
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

@app.get("/metrics")
def metrics():
    return PlainTextResponse(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    model_name = req.model
    INFLIGHT.labels(model=model_name).inc()
    try:
        rid = f"chatcmpl_{uuid.uuid4().hex[:24]}"

        # Try to load a real model unless echo is forced
        bundle  = REGISTRY.load(model_name)

        # real model path (after bundle is loaded)
        tok = bundle.tokenizer
        prompt = render_messages(tok, req.messages)
        eot = tok.convert_tokens_to_ids("<|eot_id|>")
        eos = tok.eos_token_id
        eos_ids = [i for i in (eot, eos) if i is not None]
        enc = tok(prompt, return_tensors="pt", add_special_tokens=False)

        outq: asyncio.Queue = asyncio.Queue()

        # sampler config
        cfg = SamplerCfg(
            temperature=float(req.temperature if req.temperature is not None else 1.0),
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

        greq = GenRequest(
            model_name=req.model,
            prompt=prompt,
            tok=tok,
            max_new=int(req.max_tokens or 16),
            cfg=cfg,
            eos_ids=eos_ids,
            seed=req.seed,
            outq=outq,
            want_logprobs=bool(req.logprobs),
            top_logprobs=int(req.top_logprobs or 0),
        )
        await SCHED.submit(greq)

        if req.stream:
            async def stream():
                yield _sse_chunk(rid, req.model, role="assistant")
                while True:
                    item = await outq.get()
                    if item is None:
                        break
                    if isinstance(item, dict):  # contains piece + logprobs
                        yield _sse_chunk(
                            rid, req.model,
                            content=item["piece"],
                            extra={"logprobs": item["logprobs"]}
                        )
                    else:
                        yield _sse_chunk(rid, req.model, content=item)
                yield _sse_done(rid, req.model)
            return StreamingResponse(stream(), media_type="text/event-stream")

        logprob_tokens = []
        while True:
            item = await outq.get()
            if item is None:
                break
            if isinstance(item, dict):
                logprob_tokens.extend(item["logprobs"]["content"])

        # Build final text from token IDs (robust w/ BPE)
        prefix_len = greq.input_ids.size(1) if greq.input_ids is not None else 0
        gen_ids = greq.generated[0, prefix_len:]
        text = tok.decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        payload = _non_stream_payload(rid, req.model, text)
        if req.logprobs:
            payload["logprobs"] = {"content": logprob_tokens}
        return JSONResponse(payload)
    finally:
        INFLIGHT.labels(model=model_name).dec()


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
