"""Utility functions for model management."""
import os
import logging
from pathlib import Path
from typing import Optional

try:
    from huggingface_hub import snapshot_download, hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False

from ..core import config_manager

logger = logging.getLogger(__name__)


def download_model(
    repo_id: str,
    local_dir: Optional[str] = None,
    revision: Optional[str] = None
) -> str:
    """
    Download a model from Hugging Face Hub.
    
    Args:
        repo_id: Repository ID on Hugging Face Hub
        local_dir: Local directory to save the model
        revision: Specific revision to download
        
    Returns:
        Path to the downloaded model
    """
    if not HF_HUB_AVAILABLE:
        raise ImportError("huggingface_hub is not installed. Run: pip install huggingface_hub")
    
    if local_dir is None:
        local_dir = config_manager.get_model_path()
    
    try:
        logger.info(f"Downloading model {repo_id} to {local_dir}")
        
        # Download the model
        downloaded_path = snapshot_download(
            repo_id=repo_id,
            local_dir=local_dir,
            revision=revision,
            ignore_patterns=["*.git*", "README.md", "*.txt"]
        )
        
        logger.info(f"Model downloaded successfully to {downloaded_path}")
        return downloaded_path
        
    except Exception as e:
        logger.error(f"Failed to download model {repo_id}: {e}")
        raise


def check_model_exists(model_path: str) -> bool:
    """
    Check if a model exists locally.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        True if model exists, False otherwise
    """
    path = Path(model_path)
    if not path.exists():
        return False
    
    # Check for essential OpenVINO model files
    required_files = [
        "openvino_model.xml",
        "openvino_model.bin"
    ]
    
    for file_name in required_files:
        if not (path / file_name).exists():
            logger.warning(f"Required file not found: {file_name}")
            return False
    
    return True


def get_model_size(model_path: str) -> int:
    """
    Get the total size of a model directory in bytes.
    
    Args:
        model_path: Path to the model directory
        
    Returns:
        Size in bytes
    """
    total_size = 0
    path = Path(model_path)
    
    if not path.exists():
        return 0
    
    for file_path in path.rglob("*"):
        if file_path.is_file():
            total_size += file_path.stat().st_size
    
    return total_size


def format_size(size_bytes: int) -> str:
    """
    Format size in bytes to human readable format.
    
    Args:
        size_bytes: Size in bytes
        
    Returns:
        Formatted size string
    """
    if size_bytes == 0:
        return "0 B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f} {size_names[i]}"


def cleanup_model_cache(model_path: str, keep_latest: int = 1) -> None:
    """
    Clean up old model versions, keeping only the latest ones.
    
    Args:
        model_path: Path to the model directory
        keep_latest: Number of latest versions to keep
    """
    logger.info(f"Cleaning up model cache in {model_path}")
    
    # This is a placeholder for cache cleanup logic
    # In practice, you might want to implement version-based cleanup
    pass