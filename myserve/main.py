import asyncio
import json
import time
import uuid
from typing import AsyncGenerator

from fastapi import FastAPI, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse

from myserve.api.openai_types import ChatCompletionRequest
from myserve.core.tokenizer import get_tokenizer, render_messages

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

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    # Build the prompt from chat messages
    prompt = render_messages(req.messages)
    tokenizer = get_tokenizer(req.model)

    # Token-echo: turn the *prompt tokens* into the assistant's output tokens
    input_ids = tokenizer.encode(prompt, add_special_tokens=False)

    # Respect max_tokens by truncating the echo
    max_toks = max(0, int(req.max_tokens or 0))
    if max_toks:
        output_ids = input_ids[:max_toks]
    else:
        # default: cap to 128 to avoid huge responses if user pasted a novel
        output_ids = input_ids[:128]

    created = int(time.time())
    model_name = req.model
    rid = f"chatcmpl_{uuid.uuid4().hex[:24]}"

    if req.stream:
        async def event_stream() -> AsyncGenerator[bytes, None]:
            # first chunk has the role field
            preamble = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }],
            }
            yield f"data: {json.dumps(preamble, ensure_ascii=False)}\n\n".encode()

            # stream token-by-token as delta.content
            for tid in output_ids:
                piece = tokenizer.decode([tid], skip_special_tokens=True, clean_up_tokenization_spaces=False)
                if piece == "":
                    continue
                chunk = {
                    "id": rid,
                    "object": "chat.completion.chunk",
                    "created": created,
                    "model": model_name,
                    "choices": [{
                        "index": 0,
                        "delta": {"content": piece},
                        "finish_reason": None,
                    }],
                }
                yield f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n".encode()
                # tiny delay to make streaming visible in demos
                await asyncio.sleep(0.0)

            # finalizer chunk
            final = {
                "id": rid,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [{
                    "index": 0,
                    "delta": {},
                    "finish_reason": "stop",
                }],
            }
            yield f"data: {json.dumps(final, ensure_ascii=False)}\n\n".encode()
            yield b"data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming: assemble the whole string
    text = tokenizer.decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    payload = {
        "id": rid,
        "object": "chat.completion",
        "created": created,
        "model": model_name,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop",
        }],
    }
    return JSONResponse(payload)