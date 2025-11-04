"""
Unsloth-based containers for memory-efficient inference on T4 GPUs.

Unsloth provides:
- 2x faster inference than standard transformers
- 60% less memory usage
- Perfect for T4 GPUs (cheapest Modal GPU!)
- Supports: Llama, Mistral, Phi, Gemma, Qwen

Cost comparison (Modal pricing):
- T4 (16GB):  ~$0.60/hour  ← Cheapest!
- A10G (24GB): ~$1.10/hour
- A100 (40GB): ~$3.00/hour
"""

import os
from pathlib import Path

import modal
import sentry_sdk

from runner.engines.unsloth import UnslothEngine, UnslothParams, unsloth_image
from runner.shared.common import stub
from shared.config import is_env_dev
from shared.logging import (
    get_logger,
    get_observability_secrets,
)
from shared.volumes import (
    does_model_exist,
    get_model_path,
    models_path,
    models_volume,
)

# Create mount for modal directory (same as in __init__.py)
# Modal 0.64+: Use modal.Mount instead of importing Mount
modal_path = Path(__file__).parent.parent.parent
code_mount = modal.Mount.from_local_dir(modal_path, remote_path="/root")


def _make_unsloth_container(
    name: str,
    model_name: str,
    gpu: str = "T4",  # T4 is cheapest!
    gpu_count: int = 1,
    max_seq_length: int = 2048,
    load_in_4bit: bool = True,  # 4-bit for T4
    concurrent_inputs: int = 2,  # Lower for T4
    max_containers: int = None,
    container_idle_timeout: int = 10 * 60,  # 10 minutes (save costs)
    keep_warm: int = None,
):
    """
    Create an Unsloth-powered container optimized for T4 GPUs.

    Args:
        name: Container name
        model_name: HuggingFace model ID
        gpu: GPU type (default: "T4" - cheapest option!)
        gpu_count: Number of GPUs (usually 1 for T4)
        max_seq_length: Maximum sequence length
        load_in_4bit: Use 4-bit quantization (recommended for T4)
        concurrent_inputs: Number of concurrent requests
        max_containers: Maximum number of containers
        container_idle_timeout: Idle timeout in seconds
        keep_warm: Number of containers to keep warm
    """

    # Avoid wasting resources & money in dev
    if keep_warm and is_env_dev():
        print(f"Dev environment detected, disabling keep_warm for {name}")
        keep_warm = None

    class _UnslothContainer(UnslothEngine):
        def __init__(self):
            logger = get_logger(name)
            try:
                model_path = get_model_path(model_name=model_name)
                if not does_model_exist(model_path):
                    raise Exception(f"Unable to locate model {model_path}")

                super().__init__(
                    params=UnslothParams(
                        model=str(model_path),
                        max_seq_length=max_seq_length,
                        load_in_4bit=load_in_4bit,
                        trust_remote_code=True,
                    ),
                )

                logger.info(f"Unsloth container initialized for {model_name}")

            except Exception as e:
                sentry_sdk.capture_exception(e)
                logger.exception(
                    "Failed to initialize Unsloth engine",
                    extra={"model": str(model_path)},
                )
                raise e

    _UnslothContainer.__name__ = name

    wrap = stub.cls(
        mounts=[code_mount],  # Mount modal directory for shared/ access
        volumes={models_path: models_volume},
        image=unsloth_image,
        memory=2048,  # 2GB should be enough for T4
        gpu=gpu,
        allow_concurrent_inputs=concurrent_inputs,
        container_idle_timeout=container_idle_timeout,
        timeout=10 * 60,
        secrets=[*get_observability_secrets()],
        concurrency_limit=max_containers,
        keep_warm=keep_warm,
        serialized=True,
    )

    _cls = wrap(_UnslothContainer)
    UNSLOTH_CONTAINERS[model_name] = _cls
    return _cls


# A mapping of model names to their respective Unsloth container classes
UNSLOTH_CONTAINERS = {}


# ============================================================================
# T4-Optimized Models (Cheapest GPU!)
# ============================================================================

# Phi-2: 2.7B parameter model - perfect for T4
# Very fast, low memory, great quality for small tasks
_phi2_unsloth = "microsoft/phi-2"
UnslothContainer_Phi2 = _make_unsloth_container(
    name="UnslothContainer_Phi2",
    model_name=_phi2_unsloth,
    gpu="T4",  # Only $0.60/hour!
    max_seq_length=2048,
    load_in_4bit=True,
    concurrent_inputs=3,
    max_containers=10,
    container_idle_timeout=5 * 60,  # 5 min to save costs
)

# Llama-3.2-3B: Latest Llama model, runs great on T4
# Good quality, efficient, general purpose
_llama32_3b = "unsloth/Llama-3.2-3B-Instruct"
UnslothContainer_Llama32_3B = _make_unsloth_container(
    name="UnslothContainer_Llama32_3B",
    model_name=_llama32_3b,
    gpu="T4",
    max_seq_length=4096,
    load_in_4bit=True,
    concurrent_inputs=2,
    max_containers=10,
)

# Mistral-7B: Popular 7B model optimized by Unsloth
# Great quality, fits on T4 with 4-bit quantization
_mistral_7b = "unsloth/mistral-7b-instruct-v0.3"
UnslothContainer_Mistral7B = _make_unsloth_container(
    name="UnslothContainer_Mistral7B",
    model_name=_mistral_7b,
    gpu="T4",
    max_seq_length=4096,
    load_in_4bit=True,
    concurrent_inputs=2,
    max_containers=8,
)

# Gemma-2B: Google's efficient 2B model
# Ultra fast, very cheap to run
_gemma_2b = "unsloth/gemma-2b-it"
UnslothContainer_Gemma2B = _make_unsloth_container(
    name="UnslothContainer_Gemma2B",
    model_name=_gemma_2b,
    gpu="T4",
    max_seq_length=2048,
    load_in_4bit=True,
    concurrent_inputs=4,
    max_containers=15,
    container_idle_timeout=3 * 60,  # 3 min - very cheap
)

# Qwen-2.5-3B: Latest Qwen model, excellent quality
# Great for coding and multilingual tasks
_qwen_3b = "unsloth/Qwen2.5-3B-Instruct"
UnslothContainer_Qwen3B = _make_unsloth_container(
    name="UnslothContainer_Qwen3B",
    model_name=_qwen_3b,
    gpu="T4",
    max_seq_length=4096,
    load_in_4bit=True,
    concurrent_inputs=2,
    max_containers=10,
)


# Model name mapping for easier access
# Map friendly names to actual model IDs
UNSLOTH_MODEL_MAP = {
    # Phi models
    "phi-2": _phi2_unsloth,
    "microsoft/phi-2": _phi2_unsloth,

    # Llama models
    "llama-3.2-3b": _llama32_3b,
    "llama-3.2-3b-instruct": _llama32_3b,

    # Mistral models
    "mistral-7b": _mistral_7b,
    "mistral-7b-instruct": _mistral_7b,

    # Gemma models
    "gemma-2b": _gemma_2b,
    "gemma-2b-it": _gemma_2b,

    # Qwen models
    "qwen-3b": _qwen_3b,
    "qwen-2.5-3b": _qwen_3b,
    "qwen-2.5-3b-instruct": _qwen_3b,
}
