"""Minimal logging setup helper."""

from __future__ import annotations

import logging


def configure_logging(level: int = logging.INFO) -> None:
    """Configure a consistent application-wide logging format.

    This helper centralizes the log format used across notebooks and scripts so
    debugging output stays readable regardless of the entry point.
    """

    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
