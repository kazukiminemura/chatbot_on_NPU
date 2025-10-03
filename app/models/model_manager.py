"""Model manager for handling model lifecycle."""
import asyncio
import os
import psutil
from typing import Optional, Dict, Any
from ..core import config_manager, logger
from .ov_inference import OpenVINOInferenceEngine


class ModelManager:
    """Manages model lifecycle and provides inference interface."""
    
    def __init__(self):
        self.inference_engine: Optional[OpenVINOInferenceEngine] = None
        self.config = config_manager.config
        self.is_initialized = False
        
    async def initialize(self) -> bool:
        """Initialize the model manager and load the model."""
        try:
            logger.info("Initializing model manager...")
            
            # Check system resources
            if not self._check_system_requirements():
                return False
            
            # Initialize inference engine
            self.inference_engine = OpenVINOInferenceEngine()
            success = await self.inference_engine.initialize()
            
            if success:
                self.is_initialized = True
                logger.info("Model manager initialized successfully")
                return True
            else:
                logger.error("Failed to initialize inference engine")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing model manager: {e}")
            return False
    
    def _check_system_requirements(self) -> bool:
        """Check if system meets minimum requirements."""
        try:
            # Check memory
            memory = psutil.virtual_memory()
            available_gb = memory.available / (1024**3)
            
            if available_gb < 4:
                logger.warning(f"Low memory: {available_gb:.1f}GB available. Recommended: 4GB+")
            
            # Check disk space
            model_dir = config_manager.get_model_path()
            disk_usage = psutil.disk_usage(os.path.dirname(model_dir))
            free_gb = disk_usage.free / (1024**3)
            
            if free_gb < 2:
                logger.error(f"Insufficient disk space: {free_gb:.1f}GB free. Required: 2GB+")
                return False
            
            logger.info(f"System check passed. Memory: {available_gb:.1f}GB, Disk: {free_gb:.1f}GB")
            return True
            
        except Exception as e:
            logger.error(f"Error checking system requirements: {e}")
            return False
    
    async def generate_response(self, prompt: str, **kwargs):
        """Generate a streaming response."""
        if not self.is_initialized or not self.inference_engine:
            raise RuntimeError("Model manager not initialized")
        
        async for token in self.inference_engine.generate_response(prompt, **kwargs):
            yield token
    
    async def generate_single_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """Generate a single non-streaming response."""
        if not self.is_initialized or not self.inference_engine:
            raise RuntimeError("Model manager not initialized")
        
        return await self.inference_engine.generate_single_response(prompt, **kwargs)
    
    def get_status(self) -> Dict[str, Any]:
        """Get the current status of the model manager."""
        status = {
            "initialized": self.is_initialized,
            "model_loaded": self.inference_engine.is_loaded if self.inference_engine else False,
        }
        
        if self.inference_engine:
            status.update(self.inference_engine.get_model_info())
        
        # Add system info
        memory = psutil.virtual_memory()
        status.update({
            "memory_usage_gb": (memory.total - memory.available) / (1024**3),
            "memory_available_gb": memory.available / (1024**3),
            "cpu_percent": psutil.cpu_percent(interval=1),
        })
        
        return status
    
    async def reload_model(self) -> bool:
        """Reload the model with current configuration."""
        logger.info("Reloading model...")
        
        if self.inference_engine:
            self.inference_engine = None
        
        self.is_initialized = False
        return await self.initialize()


# Global model manager instance
model_manager = ModelManager()