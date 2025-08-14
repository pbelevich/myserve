from typing import List, Optional, Literal
from pydantic import BaseModel, ConfigDict

Role = Literal["system", "user", "assistant", "tool"]

class ChatMessage(BaseModel):
    role: Role
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    # sampling
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    top_k: Optional[int] = None           # OpenAI doesn’t expose top_k, but we support it
    n: Optional[int] = 1
    max_tokens: Optional[int] = 256
    seed: Optional[int] = None            # not in OpenAI; helpful for tests
    # penalties (OpenAI compatible semantics)
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    # logprobs
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = 0       # 0 → only chosen token
    # streaming
    stream: Optional[bool] = False
    user: Optional[str] = None

    model_config = ConfigDict(extra="ignore")
