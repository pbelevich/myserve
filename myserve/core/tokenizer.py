from functools import lru_cache
from transformers import AutoTokenizer
from typing import Iterable

@lru_cache(maxsize=8)
def get_tokenizer(model_name: str):
    """Load and cache a fast tokenizer. Defaults to gpt2 if a model is missing."""
    try:
        tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception:
        tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
    # ensure consistent behavior while streaming one token at a time
    tok.padding_side = "left"
    return tok

def render_messages(messages: Iterable):
    """Flatten chat messages into a single prompt string.
    This is deliberately simple for post #1 and will evolve later.
    """
    parts = []
    for m in messages:
        if m.role == "system":
            parts.append(m.content.strip())
        elif m.role == "user":
            parts.append(m.content.strip())
        elif m.role == "assistant":
            parts.append(m.content.strip())
        # tool messages ignored in post #1
    return "\n".join(p for p in parts if p).strip()
