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

# Create mount for modal directory (same as in __init__.py)
modal_path = Path(__file__).parent.parent.parent
code_mount = modal.Mount.from_local_dir(modal_path, remote_path="/root")

# FAST_BOOT mode: Trade startup time for inference performance
# - True (dev): Faster startup, slower inference (no CUDA graph compilation)
# - False (prod): Slower startup, faster inference (full CUDA graph compilation)
FAST_BOOT = is_env_dev()


def _make_container(
    name: str,
    model_name: str,
    gpu: str = "T4",  # Default to T4 (cheapest option!)
    gpu_count: int = 1,
    concurrent_inputs: int = 16,  # Reduced for T4 (less VRAM)
    max_containers: int = None,
    container_idle_timeout: int = 10 * 60,  # 10 minutes (save costs on T4)
    keep_warm: int = None,
    **vllm_opts,
):
    """Helper function to create a container with the given GPU configuration.

    Modal 0.64+ GPU specification:
    - gpu: String like "T4", "A10G", "A100", "any", or "H100"
    - gpu_count: Number of GPUs for tensor parallelism
    - T4 is the cheapest GPU (~$0.60/hour) perfect for GPTQ quantized models
    """

    num_gpus = gpu_count

    # Avoid wasting resources & money in dev
    if keep_warm and is_env_dev():
        print("Dev environment detected, disabling keep_warm for", name)
        keep_warm = None

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

    wrap = stub.cls(
        mounts=[code_mount],  # Mount modal directory for shared/ access
        volumes={
            models_path: models_volume,
            vllm_cache_path: vllm_cache_volume,  # Cache JIT compilation artifacts
        },
        image=vllm_image,
        # Default CPU memory is 128 on modal. Request more memory for larger
        # windows of vLLM's batch loading weights into GPU memory.
        memory=1024,
        gpu=gpu,
        allow_concurrent_inputs=concurrent_inputs,
        container_idle_timeout=container_idle_timeout,
        timeout=10 * 60,
        secrets=[*get_observability_secrets()],
        concurrency_limit=max_containers,
        keep_warm=keep_warm,
        serialized=True,  # Modal 0.64+: Required for classes defined in function scope
    )
    _cls = wrap(_VllmContainer)
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
    max_containers=10,  # More containers since T4 is cheaper
    container_idle_timeout=5 * 60,  # 5 min - save costs
    quantization="GPTQ",
)

_neural_chat = "TheBloke/neural-chat-7b-v3-1-GPTQ"
VllmContainer_IntelNeuralChat7B = _make_container(
    name="VllmContainer_IntelNeuralChat7B",
    model_name=_neural_chat,
    gpu="T4",  # T4 ($0.60/hour) - GPTQ 7B fits in 16GB
    gpu_count=1,
    concurrent_inputs=12,  # T4 optimized: Medium model
    max_containers=8,
    container_idle_timeout=5 * 60,  # 5 min - save costs
    quantization="GPTQ",
)

_psyfighter2 = "TheBloke/LLaMA2-13B-Psyfighter2-GPTQ"
VllmContainer_KoboldAIPsyfighter2 = _make_container(
    name="VllmContainer_KoboldAIPsyfighter2",
    model_name=_psyfighter2,
    gpu="T4",  # T4 ($0.60/hour) - GPTQ 13B fits with careful memory management
    gpu_count=1,
    concurrent_inputs=8,  # T4 optimized: Larger model, lower concurrency
    max_containers=6,
    container_idle_timeout=5 * 60,  # 5 min - save costs
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
