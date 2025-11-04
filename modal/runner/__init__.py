from pathlib import Path
import modal
from modal import Secret, asgi_app

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

# Modal 1.0: Add local source code to image instead of using Mount
# This replaces the deprecated mount= parameter
modal_path = Path(__file__).parent.parent
completion_image = BASE_IMAGE.add_local_dir(
    str(modal_path),
    remote_path="/root",
    copy=False  # Mount at runtime for faster dev iteration
)


@stub.function(
    image=completion_image,  # Image now includes source code
    secrets=[
        Secret.from_name("ext-api-key"),
        *get_observability_secrets(),
    ],
    timeout=60 * 15,
    volumes={models_path: models_volume},
    cpu=2,
    memory=1024,
)
@modal.concurrent(max_inputs=100)  # Modal 1.0: Replaced allow_concurrent_inputs
@asgi_app()
def completion():  # named for backwards compatibility with the Modal URL
    from .api import api_app

    return api_app


# Modal 1.0: Add source to downloader image too
downloader_image_with_source = downloader_image.add_local_dir(
    str(modal_path),
    remote_path="/root",
    copy=False
)

@stub.function(
    image=downloader_image_with_source,  # Image now includes source code
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
    image=completion_image,  # Reuse completion_image (already has source)
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
