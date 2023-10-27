from .run import run as default_run
from .interval_run import run as interval_run

REGISTRY = {}
REGISTRY["default"] = default_run
REGISTRY["interval_run"] = interval_run