"""Configuration management for the chatbot application."""
import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import BaseModel


class ModelConfig(BaseModel):
    name: str
    repo_id: str
    model_type: str
    max_context_length: int
    precision: str


class InferenceConfig(BaseModel):
    max_tokens: int
    temperature: float
    top_p: float
    top_k: int
    repetition_penalty: float
    do_sample: bool


class ServerConfig(BaseModel):
    host: str
    port: int
    log_level: str


class HardwareConfig(BaseModel):
    device: str
    precision: str
    batch_size: int


class OpenVINOConfig(BaseModel):
    compile_config: Dict[str, str]


class AppConfig(BaseModel):
    model: ModelConfig
    inference: InferenceConfig
    server: ServerConfig
    hardware: HardwareConfig
    openvino: OpenVINOConfig


class ConfigManager:
    """Configuration manager for the application."""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or self._get_default_config_path()
        self._config: Optional[AppConfig] = None
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        return os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
    
    def load_config(self) -> AppConfig:
        """Load configuration from file."""
        if self._config is None:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_dict = json.load(f)
            self._config = AppConfig(**config_dict)
        return self._config
    
    def get_model_path(self) -> str:
        """Get the model storage path."""
        config = self.load_config()
        base_dir = os.path.dirname(self.config_path)
        model_dir = os.path.join(base_dir, "models", config.model.name)
        os.makedirs(model_dir, exist_ok=True)
        return model_dir
    
    def get_logs_path(self) -> str:
        """Get the logs directory path."""
        base_dir = os.path.dirname(self.config_path)
        logs_dir = os.path.join(base_dir, "logs")
        os.makedirs(logs_dir, exist_ok=True)
        return logs_dir
    
    @property
    def config(self) -> AppConfig:
        """Get the current configuration."""
        return self.load_config()


# Global config instance
config_manager = ConfigManager()