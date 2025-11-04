from pathlib import Path
from modal import Secret, asgi_app, Mount

# Import app first to register it
from runner.shared.common import app

# Keep stub as alias for backward compatibility
stub = app

# Now import everything else
from runner.containers.vllm_unified import REGISTERED_CONTAINERS
from runner.containers.unsloth_unified import UNSLOTH_CONTAINERS  # T4-optimized containers
from runner.shared.clean import clean_models_volume
from runner.shared.download import download_model, downloader_image
from shared.images import BASE_IMAGE
from shared.logging import get_logger, get_observability_secrets
from shared.volumes import models_path, models_volume


# Combine all container registries
ALL_CONTAINERS = {**REGISTERED_CONTAINERS, **UNSLOTH_CONTAINERS}

# Create mount for the modal directory to include both runner/ and shared/
# This ensures shared/ is accessible in Modal containers
modal_path = Path(__file__).parent.parent
code_mount = Mount.from_local_dir(modal_path, remote_path="/root")


@stub.function(
    image=BASE_IMAGE,
    mounts=[code_mount],  # Mount modal directory for shared/ access
    secrets=[
        Secret.from_name("ext-api-key"),
        *get_observability_secrets(),
    ],
    timeout=60 * 15,
    allow_concurrent_inputs=100,
    volumes={models_path: models_volume},
    cpu=2,
    memory=1024,
)
@asgi_app()
def completion():  # named for backwards compatibility with the Modal URL
    from .api import api_app

    return api_app


@stub.function(
    image=downloader_image,
    mounts=[code_mount],  # Mount modal directory for shared/ access
    timeout=3600,  # 1 hour
    volumes={models_path: models_volume},
    secrets=[
        Secret.from_name("huggingface"),
        *get_observability_secrets(),
    ],
)
def download(force: bool = False):
    logger = get_logger("download")
    logger.info("Downloading all models...")
    for model in ALL_CONTAINERS:  # Updated to use ALL_CONTAINERS
        # Can't be parallelized because of a modal volume corruption issue
        download_model.local(model, force=force)
    logger.info("ALL DONE!")


@stub.function(
    image=BASE_IMAGE,
    mounts=[code_mount],  # Mount modal directory for shared/ access
    volumes={models_path: models_volume},
    secrets=[
        *get_observability_secrets(),
    ],
)
def clean(all: bool = False, dry: bool = False):
    logger = get_logger("clean")
    logger.info(f"Cleaning models volume. ALL: {all}. DRY: {dry}")
    remaining_models = [] if all else [m.lower() for m in ALL_CONTAINERS]  # Updated
    clean_models_volume(remaining_models, dry)
