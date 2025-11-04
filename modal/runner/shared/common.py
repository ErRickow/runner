from modal import App

from shared.config import Config

config = Config(
    name="runner",
    api_key_id="RUNNER_API_KEY",
)

BACKLOG_THRESHOLD = 30

# Modal 0.64+: Stub renamed to App
app = App(config.name)

# Backward compatibility alias
stub = app
