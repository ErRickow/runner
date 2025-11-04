from pathlib import Path
from modal import App, Mount

from shared.config import Config

config = Config(
    name="runner",
    api_key_id="RUNNER_API_KEY",
)

BACKLOG_THRESHOLD = 30

# Modal 0.64+: Stub renamed to App
# Add mount for the entire modal directory to ensure shared/ is accessible
modal_dir = Path(__file__).parent.parent.parent
app = App(
    name=config.name,
)

# Backward compatibility alias
stub = app
