import time
from typing import Optional, Union

from modal import Image, enter, method
from pydantic import BaseModel

from shared.logging import (
    add_observability,
    get_logger,
    timer,
)
from shared.protocol import (
    CompletionPayload,
    CompletionRequest,
    CompletionResponse,
    CompletionChoice,
    CompletionChunk,
    Usage,
    create_error_text,
    generate_completion_id,
    get_current_timestamp,
    sse,
    sse_done,
)

from .base import BaseEngine

logger = get_logger(__name__)


# Updated to vLLM 0.10.2 with latest optimizations
# Using CUDA 12.4 for better compatibility with latest vLLM
# Torch 2.4.1 is stable and compatible with vLLM 0.10.2
vllm_image = add_observability(
    Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install(
        "torch==2.4.1",  # Stable version compatible with vLLM 0.10.2
        "vllm==0.10.2",
        "sentry-sdk==2.17.0",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})  # Faster model downloads
)

with vllm_image.imports():
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine


# Adapted from: https://github.com/vllm-project/vllm/blob/main/vllm/engine/arg_utils.py#L192
class VllmParams(BaseModel):
    model: str
    tokenizer: Optional[str] = None
    tokenizer_mode: str = "auto"
    trust_remote_code: bool = True
    download_dir: Optional[str] = None
    load_format: str = "auto"
    dtype: str = "auto"
    seed: int = 0
    max_model_len: Optional[int] = None
    worker_use_ray: bool = False
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    block_size: int = 16
    swap_space: int = 4  # GiB
    gpu_memory_utilization: float = 0.90
    max_num_batched_tokens: Optional[int] = None
    max_num_seqs: int = 256
    # max_paddings: int = 256
    disable_log_stats: bool = False
    revision: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    quantization: Optional[str] = None
    enforce_eager: Optional[bool] = None  # FAST_BOOT mode support


class VllmEngine(BaseEngine):
    def __init__(self, params: VllmParams):
        self.engine = None
        self.engine_args = AsyncEngineArgs(
            **params.dict(),
            disable_log_requests=True,
        )

    @enter()
    def startup(self):
        with timer("engine init", model=self.engine_args.model):
            self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)

    @method()
    async def generate(
        self,
        payload: Union[CompletionPayload, CompletionRequest],
        params,
    ):
        """Generate completion using vLLM engine with OpenAI-compatible format."""
        assert self.engine is not None, "Engine not initialized"

        # Handle both legacy and new request formats
        if isinstance(payload, CompletionPayload):
            # Legacy format - convert to new format
            request_id = payload.id
            prompt = payload.prompt
            stream = payload.stream
            model_name = payload.model
        else:
            # New OpenAI-compatible format
            request_id = generate_completion_id()
            prompt = payload.prompt if isinstance(payload.prompt, str) else payload.prompt[0]
            stream = payload.stream
            model_name = payload.model

        t_start_inference = time.time()
        created_timestamp = get_current_timestamp()

        try:
            results_generator = self.engine.generate(
                prompt, params, request_id
            )

            output = ""
            index = 0
            finish_reason = None
            prompt_tokens = 0
            completion_tokens = 0

            async for current in results_generator:
                output = current.outputs[0].text
                finish_reason = current.outputs[0].finish_reason
                prompt_tokens = len(current.prompt_token_ids)
                completion_tokens = len(current.outputs[0].token_ids)

                # Non-streaming requests continue generating w/o yielding intermediate results
                if not stream:
                    yield " "  # HACK: Keep the connection alive while generating
                    continue

                # Skipping invalid UTF8 tokens:
                if output and output[-1] == "\ufffd":
                    continue

                # Streaming: send incremental token in OpenAI format
                token = output[index:]
                index = len(output)

                chunk = CompletionChunk(
                    id=request_id,
                    created=created_timestamp,
                    model=model_name,
                    choices=[
                        CompletionChoice(
                            text=token,
                            index=0,
                            logprobs=None,
                            finish_reason=finish_reason,
                        )
                    ],
                )
                yield sse(chunk.model_dump_json())

            # Send final response
            if stream:
                # For streaming, send final chunk with finish_reason
                final_chunk = CompletionChunk(
                    id=request_id,
                    created=created_timestamp,
                    model=model_name,
                    choices=[
                        CompletionChoice(
                            text="",
                            index=0,
                            logprobs=None,
                            finish_reason=finish_reason or "stop",
                        )
                    ],
                )
                yield sse(final_chunk.model_dump_json())
                yield sse_done()
            else:
                # For non-streaming, send complete response
                response = CompletionResponse(
                    id=request_id,
                    created=created_timestamp,
                    model=model_name,
                    choices=[
                        CompletionChoice(
                            text=output,
                            index=0,
                            logprobs=None,
                            finish_reason=finish_reason or "stop",
                        )
                    ],
                    usage=Usage(
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        total_tokens=prompt_tokens + completion_tokens,
                    ),
                )
                yield response.model_dump_json()

            duration = time.time() - t_start_inference
            logger.info(
                "Completed generation",
                extra={
                    "model": self.engine_args.model,
                    "tokens": completion_tokens,
                    "tps": completion_tokens / duration if duration > 0 else 0,
                    "duration": duration,
                },
            )
        except Exception as err:
            e = create_error_text(err)
            logger.exception(
                "Failed generation", extra={"model": self.engine_args.model}
            )
            yield sse(e) if stream else e
