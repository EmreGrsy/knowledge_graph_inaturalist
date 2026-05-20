"""Central logging configuration.

Use `get_logger("module_name")` to get a namespaced logger that writes to
both the console and a rotating file at `logs/app.log`. Set the `LOG_LEVEL`
environment variable (DEBUG / INFO / WARNING / ERROR) to control verbosity.
"""

import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

_LOG_DIR = Path(__file__).resolve().parent.parent / "logs"
_LOG_DIR.mkdir(exist_ok=True)
_LOG_FILE = _LOG_DIR / "app.log"

_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"

_ROOT_NAME = "bio_kg"
_configured = False


def _configure_root() -> None:
    global _configured
    if _configured:
        return

    level = os.environ.get("LOG_LEVEL", "INFO").upper()
    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    root = logging.getLogger(_ROOT_NAME)
    root.setLevel(level)
    root.propagate = False

    console = logging.StreamHandler()
    console.setFormatter(formatter)
    root.addHandler(console)

    file_handler = RotatingFileHandler(
        _LOG_FILE, maxBytes=2_000_000, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _configured = True


def get_logger(name: str) -> logging.Logger:
    """Return a namespaced logger under the `bio_kg` tree."""
    _configure_root()
    return logging.getLogger(f"{_ROOT_NAME}.{name}")
