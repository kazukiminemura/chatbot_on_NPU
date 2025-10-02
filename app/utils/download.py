"""
Model download utility for Llama2-7B NPU Chatbot
"""
import os
import logging
from pathlib import Path
from typing import Optional
from huggingface_hub import snapshot_download, HfApi
from app.core.config import config


logger = logging.getLogger(__name__)


class ModelDownloader:
    """Model downloader class"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(exist_ok=True)
        self.api = HfApi()
        
    def download_model(self, repo_id: str, model_name: str, force_download: bool = False) -> Path:
        """
        Download model from HuggingFace Hub
        
        Args:
            repo_id: HuggingFace repository ID
            model_name: Local model name
            force_download: Force re-download even if model exists
            
        Returns:
            Path to downloaded model directory
        """
        model_path = self.models_dir / model_name
        
        if model_path.exists() and not force_download:
            logger.info(f"Model already exists at {model_path}")
            return model_path
            
        logger.info(f"Downloading model {repo_id} to {model_path}")
        
        try:
            # Check if model exists on HuggingFace Hub
            model_info = self.api.model_info(repo_id)
            logger.info(f"Model info: {model_info.modelId}, size: ~{model_info.safetensors}")
            
            # Download model
            downloaded_path = snapshot_download(
                repo_id=repo_id,
                local_dir=model_path,
                local_dir_use_symlinks=False,
                resume_download=True
            )
            
            logger.info(f"Model downloaded successfully to {downloaded_path}")
            return Path(downloaded_path)
            
        except Exception as e:
            logger.error(f"Failed to download model {repo_id}: {str(e)}")
            raise
    
    def check_model_exists(self, model_name: str) -> bool:
        """Check if model exists locally"""
        model_path = self.models_dir / model_name
        return model_path.exists() and any(model_path.iterdir())
    
    def get_model_path(self, model_name: str) -> Path:
        """Get path to model directory"""
        return self.models_dir / model_name
    
    def list_downloaded_models(self) -> list:
        """List all downloaded models"""
        models = []
        for item in self.models_dir.iterdir():
            if item.is_dir() and any(item.iterdir()):
                models.append(item.name)
        return models