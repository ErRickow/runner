from typing import Union

from fastapi import Request, status
from fastapi.responses import StreamingResponse, JSONResponse

from runner.containers.vllm_unified import (
    QUANTIZED_MODELS,
    REGISTERED_CONTAINERS,
)
from runner.shared.common import BACKLOG_THRESHOLD
from runner.shared.sampling_params import SamplingParams
from shared.logging import get_logger
from shared.protocol import (
    CompletionPayload,
    CompletionRequest,
    create_error_response,
)
from shared.volumes import does_model_exist, get_model_path

logger = get_logger(__name__)


def _get_sampling_params(payload: Union[CompletionPayload, CompletionRequest]):
    """Extract sampling parameters from either legacy or new format."""
    if isinstance(payload, CompletionPayload):
        # Legacy format: params are nested
        return SamplingParams(**payload.params.dict())
    else:
        # New OpenAI format: params are top-level
        return SamplingParams(
            max_tokens=payload.max_tokens,
            temperature=payload.temperature,
            top_p=payload.top_p,
            top_k=payload.top_k,
            n=payload.n,
            stop=payload.stop,
            presence_penalty=payload.presence_penalty,
            frequency_penalty=payload.frequency_penalty,
            repetition_penalty=payload.repetition_penalty,
            best_of=payload.best_of,
            logprobs=payload.logprobs,
            ignore_eos=payload.ignore_eos,
            use_beam_search=payload.use_beam_search,
            skip_special_tokens=payload.skip_special_tokens,
            min_p=payload.min_p,
        )


def completion(
    request: Request,
    payload: Union[CompletionPayload, CompletionRequest],
):
    """
    Handle completion requests in both legacy and OpenAI-compatible formats.

    Supports:
    - Legacy format: {id, prompt, params: {...}, model, stream}
    - OpenAI format: {model, prompt, temperature, max_tokens, ...}
    """
    # Some models are served quantized, so we try re-mapping them first
    model_name = payload.model
    if model_name in QUANTIZED_MODELS:
        model_name = QUANTIZED_MODELS[model_name]

    model_path = get_model_path(model_name)
    logger.info(
        "Received completion request",
        extra={
            "model": str(model_path),
            "format": "legacy" if isinstance(payload, CompletionPayload) else "openai",
            "user-agent": request.headers.get("user-agent"),
            "referer": request.headers.get("referer"),
            "ip": request.headers.get("x-real-ip")
            or request.headers.get("x-forwarded-for")
            or request.client.host,
        },
    )

    if not does_model_exist(model_path):
        message = f"Unable to locate model {model_name}"
        logger.error(message)
        return create_error_response(
            status.HTTP_400_BAD_REQUEST,
            message,
            "model_not_found",
        )

    container = REGISTERED_CONTAINERS.get(model_name)
    if container is None:
        message = f"Unable to locate container type for model {model_name}"
        logger.error(message)
        return create_error_response(
            status.HTTP_400_BAD_REQUEST,
            message,
            "model_not_found",
        )

    runner = container()

    stats = runner.generate.get_current_stats()
    logger.info(stats)
    if stats.backlog > BACKLOG_THRESHOLD:
        message = f"Server is currently overloaded. Please try again later."
        logger.warning(f"Backlog too high: {stats.backlog}")
        return create_error_response(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            message,
            "server_overloaded",
        )

    try:
        sampling_params = _get_sampling_params(payload)
    except ValueError as e:
        logger.exception("Invalid sampling params")
        return create_error_response(
            status.HTTP_400_BAD_REQUEST,
            str(e),
            "invalid_request_error",
        )

    async def generate():
        async for text in runner.generate.remote_gen.aio(
            payload, sampling_params
        ):
            yield text

    # For streaming, use text/event-stream
    # For non-streaming, the engine will return JSON directly
    if payload.stream:
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
        )
    else:
        # For non-streaming, we need to collect the response
        async def get_response():
            response_text = ""
            async for chunk in runner.generate.remote_gen.aio(
                payload, sampling_params
            ):
                response_text += chunk
            return response_text

        # This is handled by the engine now
        return StreamingResponse(
            generate(),
            media_type="application/json",
        )
