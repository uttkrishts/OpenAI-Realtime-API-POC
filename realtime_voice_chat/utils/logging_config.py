"""
Logging configuration for the realtime voice chat library.
"""

import logging
import sys
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    format_str: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
) -> None:
    """
    Set up logging configuration for the library.

    Args:
        level: Logging level (default: INFO)
        log_file: Optional file to write logs to
        format_str: Log message format string
    """
    # Create formatter
    formatter = logging.Formatter(format_str)

    # Create handlers
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Add new handlers
    for handler in handlers:
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    # Set library logger level
    logging.getLogger("realtime_voice_chat").setLevel(level)
