"""Core package initialization."""
from .config import config_manager
from .logger import logger, setup_logging

# Reinitialize logging with proper config once config_manager is available
try:
    config = config_manager.config
    logger = setup_logging(config)
except Exception:
    # If config loading fails, use default logger
    pass

__all__ = ["config_manager", "logger", "setup_logging"]