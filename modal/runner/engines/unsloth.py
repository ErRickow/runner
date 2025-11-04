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


# Unsloth image - optimized for T4 GPUs
# Unsloth is 2x faster and uses 60% less memory than standard inference
# Let pip install latest compatible versions automatically
unsloth_image = add_observability(
    Image.from_registry(
        "nvidia/cuda:12.1.0-base-ubuntu22.04",
        add_python="3.11",
    )
    # Step 0: Install git (required for pip install from git repos)
    .apt_install("git")
    # Step 1: Install PyTorch and core dependencies (let pip resolve versions)
    .pip_install(
        "torch",  # Latest compatible
        "torchvision",  # Latest compatible with torch
        "transformers",  # Latest
        "accelerate",  # Latest
        "bitsandbytes",  # Latest
        "xformers",  # Latest compatible with torch
    )
    # Step 2: Install Unsloth from git (pip will use compatible versions)
    .pip_install(
        "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git",
    )
    # Step 3: Add observability
    .pip_install("sentry-sdk")
)

with unsloth_image.imports():
    from unsloth import FastLanguageModel
    import torch


class UnslothParams(BaseModel):
    """Parameters for Unsloth inference engine."""
    model: str
    max_seq_length: int = 2048
    dtype: Optional[str] = None  # Auto-detect
    load_in_4bit: bool = True  # 4-bit quantization for T4
    trust_remote_code: bool = True


class UnslothEngine(BaseEngine):
    """
    Inference engine using Unsloth for memory-efficient inference on T4 GPUs.

    Benefits:
    - 2x faster than standard transformers
    - 60% less memory usage
    - Works great on T4 GPUs (cheapest option!)
    - Supports Llama, Mistral, Phi, Gemma models
    """

    def __init__(self, params: UnslothParams):
        self.params = params
        self.model = None
        self.tokenizer = None

    @enter()
    def startup(self):
        with timer("unsloth model init", model=self.params.model):
            self.model, self.tokenizer = FastLanguageModel.from_pretrained(
                model_name=self.params.model,
                max_seq_length=self.params.max_seq_length,
                dtype=self.params.dtype,
                load_in_4bit=self.params.load_in_4bit,
                trust_remote_code=self.params.trust_remote_code,
            )
            # Enable faster inference
            FastLanguageModel.for_inference(self.model)
            logger.info(f"Unsloth model loaded: {self.params.model}")

    @method()
    async def generate(
        self,
        payload: Union[CompletionPayload, CompletionRequest],
        params,
    ):
        """Generate completion using Unsloth engine."""
        assert self.model is not None, "Engine not initialized"
        assert self.tokenizer is not None, "Tokenizer not initialized"

        # Handle both legacy and new request formats
        if isinstance(payload, CompletionPayload):
            request_id = payload.id
            prompt = payload.prompt
            stream = payload.stream
            model_name = payload.model
        else:
            request_id = generate_completion_id()
            prompt = payload.prompt if isinstance(payload.prompt, str) else payload.prompt[0]
            stream = payload.stream
            model_name = payload.model

        t_start_inference = time.time()
        created_timestamp = get_current_timestamp()

        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=self.params.max_seq_length,
            ).to("cuda")

            prompt_tokens = inputs.input_ids.shape[1]

            # Generate with Unsloth
            with torch.inference_mode():
                if stream:
                    # Streaming generation
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=params.max_tokens or 100,
                        temperature=params.temperature,
                        top_p=params.top_p,
                        do_sample=params.temperature > 0,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                        use_cache=True,
                        # Streaming is done via multiple calls
                    )

                    # Decode output
                    output_text = self.tokenizer.decode(
                        outputs[0][prompt_tokens:],
                        skip_special_tokens=params.skip_special_tokens
                    )
                    completion_tokens = len(outputs[0]) - prompt_tokens

                    # Send as chunks for streaming
                    chunk_size = max(1, len(output_text) // 10)  # ~10 chunks
                    for i in range(0, len(output_text), chunk_size):
                        chunk_text = output_text[i:i + chunk_size]
                        chunk = CompletionChunk(
                            id=request_id,
                            created=created_timestamp,
                            model=model_name,
                            choices=[
                                CompletionChoice(
                                    text=chunk_text,
                                    index=0,
                                    logprobs=None,
                                    finish_reason=None if i + chunk_size < len(output_text) else "stop",
                                )
                            ],
                        )
                        yield sse(chunk.model_dump_json())

                    # Send done marker
                    final_chunk = CompletionChunk(
                        id=request_id,
                        created=created_timestamp,
                        model=model_name,
                        choices=[
                            CompletionChoice(
                                text="",
                                index=0,
                                logprobs=None,
                                finish_reason="stop",
                            )
                        ],
                    )
                    yield sse(final_chunk.model_dump_json())
                    yield sse_done()

                else:
                    # Non-streaming generation
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=params.max_tokens or 100,
                        temperature=params.temperature,
                        top_p=params.top_p,
                        do_sample=params.temperature > 0,
                        pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
                        use_cache=True,
                    )

                    # Decode output
                    output_text = self.tokenizer.decode(
                        outputs[0][prompt_tokens:],
                        skip_special_tokens=params.skip_special_tokens
                    )
                    completion_tokens = len(outputs[0]) - prompt_tokens

                    response = CompletionResponse(
                        id=request_id,
                        created=created_timestamp,
                        model=model_name,
                        choices=[
                            CompletionChoice(
                                text=output_text,
                                index=0,
                                logprobs=None,
                                finish_reason="stop",
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
                "Completed generation with Unsloth",
                extra={
                    "model": self.params.model,
                    "tokens": completion_tokens,
                    "tps": completion_tokens / duration if duration > 0 else 0,
                    "duration": duration,
                },
            )
        except Exception as err:
            e = create_error_text(err)
            logger.exception(
                "Failed generation", extra={"model": self.params.model}
            )
            yield sse(e) if stream else e
