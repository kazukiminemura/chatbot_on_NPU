"""
Configuration management module for Llama2-7B NPU Chatbot
"""
import json
import logging
from pathlib import Path
from typing import Dict, Any


class Config:
    """Configuration class for the chatbot application"""
    
    def __init__(self, config_path: str = "config.json"):
        self.config_path = Path(config_path)
        self._config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            logging.error(f"Configuration file not found: {self.config_path}")
            return self._get_default_config()
        except json.JSONDecodeError as e:
            logging.error(f"Invalid JSON in configuration file: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "model": {
                "name": "Llama2-7B",
                "repo_id": "meta-llama/Llama-2-7b-chat-hf",
                "model_type": "llama",
                "max_context_length": 4096
            },
            "inference": {
                "max_tokens": 500,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1
            },
            "server": {
                "host": "localhost",
                "port": 8000,
                "log_level": "INFO"
            },
            "hardware": {
                "device": "NPU",
                "precision": "FP16",
                "batch_size": 1
            }
        }
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by key using dot notation"""
        keys = key.split('.')
        value = self._config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration"""
        return self._config.get("model", {})
    
    def get_inference_config(self) -> Dict[str, Any]:
        """Get inference configuration"""
        return self._config.get("inference", {})
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration"""
        return self._config.get("server", {})
    
    def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware configuration"""
        return self._config.get("hardware", {})


# Global configuration instance
config = Config()