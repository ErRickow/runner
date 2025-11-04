import time
from typing import Any, Dict, List, Literal, Optional, Union

from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# OpenAI-compatible completion parameters
# Flattened to top-level as per OpenAI API standard
class CompletionRequest(BaseModel):
    """OpenAI-compatible text completion request."""
    model: str
    prompt: Union[str, List[str]]
    stream: bool = False
    max_tokens: Optional[int] = 16
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    logprobs: Optional[int] = None
    echo: bool = False
    stop: Union[None, str, List[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    best_of: Optional[int] = None
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    # Additional vLLM-specific parameters
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    ignore_eos: bool = False
    use_beam_search: bool = False
    skip_special_tokens: bool = True
    seed: Optional[int] = None


class ChatMessage(BaseModel):
    """OpenAI chat message format."""
    role: Literal["system", "user", "assistant", "function"]
    content: str
    name: Optional[str] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str
    messages: List[ChatMessage]
    stream: bool = False
    max_tokens: Optional[int] = 16
    temperature: float = 1.0
    top_p: float = 1.0
    n: int = 1
    stop: Union[None, str, List[str]] = None
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    # Additional vLLM-specific parameters
    top_k: int = -1
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    best_of: Optional[int] = None
    seed: Optional[int] = None


# Legacy format for backward compatibility
class Params(BaseModel):
    """Deprecated: Use top-level parameters instead."""
    best_of: Optional[int] = None
    ignore_eos: bool = False
    logprobs: Optional[int] = None
    max_tokens: Optional[int] = 42
    n: int = 1
    stop: Union[None, str, List[str]] = None
    temperature: float = 1.0
    top_k: int = -1
    top_p: float = 1.0
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    use_beam_search: bool = False
    skip_special_tokens: bool = True


class CompletionPayload(BaseModel):
    """Deprecated: Use CompletionRequest instead."""
    id: str
    prompt: str
    stream: bool = False
    params: Params
    model: str


# OpenAI-compatible response models
class Usage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class CompletionChoice(BaseModel):
    """Single completion choice in OpenAI format."""
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    """OpenAI-compatible text completion response."""
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: List[CompletionChoice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None


class CompletionChunk(BaseModel):
    """OpenAI-compatible streaming chunk."""
    id: str
    object: str = "text_completion.chunk"
    created: int
    model: str
    choices: List[CompletionChoice]
    system_fingerprint: Optional[str] = None


class ChatCompletionMessage(BaseModel):
    """Chat completion message in response."""
    role: str
    content: str
    function_call: Optional[Dict[str, Any]] = None


class ChatCompletionChoice(BaseModel):
    """Single chat completion choice."""
    index: int
    message: ChatCompletionMessage
    finish_reason: Optional[str] = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[ChatCompletionChoice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None


class ChatCompletionChunkDelta(BaseModel):
    """Delta content in streaming response."""
    role: Optional[str] = None
    content: Optional[str] = None


class ChatCompletionChunkChoice(BaseModel):
    """Choice in chat streaming chunk."""
    index: int
    delta: ChatCompletionChunkDelta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    """OpenAI-compatible chat streaming chunk."""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[ChatCompletionChunkChoice]
    system_fingerprint: Optional[str] = None


# Legacy response format (deprecated)
class ResponseBody(BaseModel):
    """Deprecated: Use CompletionResponse instead."""
    text: str
    usage: Usage
    finish_reason: Optional[str] = None
    done: bool = False


def sse(response: str) -> str:
    """Wrap a given response string as an SSE message."""
    return f"data: {response}\n\n"


def sse_done() -> str:
    """Return the standard SSE done message."""
    return "data: [DONE]\n\n"


# Error handling - OpenAI compatible
class ErrorDetail(BaseModel):
    """OpenAI-compatible error detail."""
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None


class ErrorResponse(BaseModel):
    """OpenAI-compatible error response."""
    error: ErrorDetail


def create_error_text(err: Exception) -> str:
    """Create error text from exception."""
    return ErrorResponse(
        error=ErrorDetail(
            message=str(err),
            type=type(err).__name__,
            param=None,
            code=None,
        )
    ).model_dump_json()


def create_error_response(status_code: int, message: str, error_type: str = "invalid_request_error") -> JSONResponse:
    """Create OpenAI-compatible error response."""
    error_response = ErrorResponse(
        error=ErrorDetail(
            message=message,
            type=error_type,
            param=None,
            code=None,
        )
    )
    return JSONResponse(
        content=error_response.model_dump(),
        status_code=status_code,
    )


def generate_completion_id() -> str:
    """Generate a unique completion ID."""
    import random
    import string
    return "cmpl-" + "".join(
        random.choices(string.ascii_letters + string.digits, k=29)
    )


def generate_chat_completion_id() -> str:
    """Generate a unique chat completion ID."""
    import random
    import string
    return "chatcmpl-" + "".join(
        random.choices(string.ascii_letters + string.digits, k=29)
    )


def get_current_timestamp() -> int:
    """Get current Unix timestamp."""
    return int(time.time())
