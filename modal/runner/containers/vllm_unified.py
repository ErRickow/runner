import os
from pathlib import Path

import modal
import modal.gpu
import sentry_sdk

from runner.engines.vllm import VllmEngine, VllmParams, vllm_image
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
    vllm_cache_volume,
    vllm_cache_path,
)

# Modal 1.0: Add local source code to vLLM image instead of using Mount
modal_path = Path(__file__).parent.parent.parent
vllm_image_with_source = vllm_image.add_local_dir(
    str(modal_path),
    remote_path="/root",
    copy=False  # Mount at runtime for faster dev iteration
)

# FAST_BOOT mode: Trade startup time for inference performance
# - True (dev): Faster startup, slower inference (no CUDA graph compilation)
# - False (prod): Slower startup, faster inference (full CUDA graph compilation)
FAST_BOOT = is_env_dev()


def _make_container(
    name: str,
    model_name: str,
    gpu: str = "T4",  # Default to T4 (cheapest option!)
    gpu_count: int = 1,
    concurrent_inputs: int = 16,  # Concurrent inputs handled by @modal.concurrent
    max_containers_limit: int = None,  # Modal 1.0: renamed from concurrency_limit
    scaledown_window: int = 10 * 60,  # Modal 1.0: renamed from container_idle_timeout
    min_containers: int = None,  # Modal 1.0: renamed from keep_warm
    **vllm_opts,
):
    """Helper function to create a container with the given GPU configuration.

    Modal 1.0 parameters:
    - gpu: String like "T4", "A10G", "A100", "any", or "H100"
    - gpu_count: Number of GPUs for tensor parallelism
    - concurrent_inputs: Number of concurrent requests (used by @modal.concurrent)
    - max_containers_limit: Max containers to spawn (renamed from concurrency_limit)
    - scaledown_window: Idle time before scaling down (renamed from container_idle_timeout)
    - min_containers: Minimum warm containers (renamed from keep_warm)
    - T4 is the cheapest GPU (~$0.60/hour) perfect for GPTQ quantized models
    """

    num_gpus = gpu_count

    # Avoid wasting resources & money in dev
    if min_containers and is_env_dev():
        print("Dev environment detected, disabling min_containers for", name)
        min_containers = None

    class _VllmContainer(VllmEngine):
        def __init__(self):
            logger = get_logger(name)
            try:
                model_path = get_model_path(model_name=model_name)
                if not does_model_exist(model_path):
                    raise Exception("Unable to locate model {}", model_path)

                if num_gpus > 1:
                    # HACK[1-20-2024]: Yesterday, Modal started populating this env var
                    # with GPU UUIDs. This breaks some assumption in Ray, so just unset
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)

                    # Patch issue from https://github.com/vllm-project/vllm/issues/1116
                    import ray

                    ray.shutdown()
                    ray.init(num_gpus=num_gpus, ignore_reinit_error=True)

                super().__init__(
                    params=VllmParams(
                        model=str(model_path),
                        tensor_parallel_size=num_gpus,
                        enforce_eager=FAST_BOOT,  # Skip CUDA compilation in dev
                        **vllm_opts,
                    ),
                )

                # Performance improvement from https://github.com/vllm-project/vllm/issues/2073#issuecomment-1853422529
                if num_gpus > 1:
                    import subprocess

                    RAY_CORE_PIN_OVERRIDE = "cpuid=0 ; for pid in $(ps xo '%p %c' | grep ray:: | awk '{print $1;}') ; do taskset -cp $cpuid $pid ; cpuid=$(($cpuid + 1)) ; done"
                    subprocess.call(RAY_CORE_PIN_OVERRIDE, shell=True)
            except Exception as e:
                # We have to manually capture and re-raise because Modal catches the exception upstream
                sentry_sdk.capture_exception(e)
                logger.exception(
                    "Failed to initialize VLLM engine",
                    extra={"model": str(model_path)},
                )
                raise e

    _VllmContainer.__name__ = name

    # Modal 1.0: Apply @modal.concurrent decorator to the class before wrapping
    cls_to_wrap = _VllmContainer
    if concurrent_inputs > 1:
        cls_to_wrap = modal.concurrent(max_inputs=concurrent_inputs)(cls_to_wrap)

    # Now apply stub.cls wrapper
    _cls = stub.cls(
        volumes={
            models_path: models_volume,
            vllm_cache_path: vllm_cache_volume,  # Cache JIT compilation artifacts
        },
        image=vllm_image_with_source,  # Modal 1.0: Image now includes source code
        # Default CPU memory is 128 on modal. Request more memory for larger
        # windows of vLLM's batch loading weights into GPU memory.
        memory=1024,
        gpu=gpu,
        scaledown_window=scaledown_window,  # Modal 1.0: renamed from container_idle_timeout
        timeout=10 * 60,
        secrets=[*get_observability_secrets()],
        max_containers=max_containers_limit,  # Modal 1.0: renamed from concurrency_limit
        min_containers=min_containers,  # Modal 1.0: renamed from keep_warm
        serialized=True,  # Modal 0.64+: Required for classes defined in function scope
    )(cls_to_wrap)

    REGISTERED_CONTAINERS[model_name] = _cls
    return _cls


# A mapping of model names to their respective container classes.
# Automatically populated by _make_container.
REGISTERED_CONTAINERS = {}

_phi2 = "TheBloke/phi-2-GPTQ"
VllmContainer_MicrosoftPhi2 = _make_container(
    name="VllmContainer_MicrosoftPhi2",
    model_name=_phi2,
    gpu="T4",  # T4 ($0.60/hour) - GPTQ quantized fits perfectly!
    gpu_count=1,
    concurrent_inputs=16,  # T4 optimized: Small model, good concurrency
    max_containers_limit=10,  # Modal 1.0: More containers since T4 is cheaper
    scaledown_window=5 * 60,  # Modal 1.0: 5 min - save costs
    quantization="GPTQ",
)

_neural_chat = "TheBloke/neural-chat-7b-v3-1-GPTQ"
VllmContainer_IntelNeuralChat7B = _make_container(
    name="VllmContainer_IntelNeuralChat7B",
    model_name=_neural_chat,
    gpu="T4",  # T4 ($0.60/hour) - GPTQ 7B fits in 16GB
    gpu_count=1,
    concurrent_inputs=12,  # T4 optimized: Medium model
    max_containers_limit=8,  # Modal 1.0
    scaledown_window=5 * 60,  # Modal 1.0: 5 min - save costs
    quantization="GPTQ",
)

_psyfighter2 = "TheBloke/LLaMA2-13B-Psyfighter2-GPTQ"
VllmContainer_KoboldAIPsyfighter2 = _make_container(
    name="VllmContainer_KoboldAIPsyfighter2",
    model_name=_psyfighter2,
    gpu="T4",  # T4 ($0.60/hour) - GPTQ 13B fits with careful memory management
    gpu_count=1,
    concurrent_inputs=8,  # T4 optimized: Larger model, lower concurrency
    max_containers_limit=6,  # Modal 1.0
    scaledown_window=5 * 60,  # Modal 1.0: 5 min - save costs
    quantization="GPTQ",
)


# A re-mapping of model names to their respective quantized models.
# From the outside, the model name is the original, but internally,
# we use the quantized model name.
#
# NOTE: When serving quantized models, the throughput can suffer a ton
#       at high batch sizes. Read this thread to learn why:
#       https://github.com/vllm-project/vllm/issues/1002#issuecomment-1712824199
QUANTIZED_MODELS = {
    "microsoft/phi-2": _phi2,
    "Intel/neural-chat-7b-v3-1": _neural_chat,
    "KoboldAI/LLaMA2-13B-Psyfighter2": _psyfighter2,
}
