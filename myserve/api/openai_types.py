from typing import List, Optional, Literal
from pydantic import BaseModel, ConfigDict

Role = Literal["system", "user", "assistant", "tool"]

class ChatMessage(BaseModel):
    role: Role
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    stream: Optional[bool] = False
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 1.0
    top_p: Optional[float] = 1.0
    n: Optional[int] = 1
    stop: Optional[list[str]] = None
    presence_penalty: Optional[float] = 0.0
    frequency_penalty: Optional[float] = 0.0
    logprobs: Optional[bool] = False
    top_logprobs: Optional[int] = None
    user: Optional[str] = None

    model_config = ConfigDict(extra="ignore")
