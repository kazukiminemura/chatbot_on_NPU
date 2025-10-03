"""Core package initialization."""
from .config import config_manager
from .logger import logger

__all__ = ["config_manager", "logger"]