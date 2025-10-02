"""
Model conversion utility for converting HuggingFace models to OpenVINO format
"""
import logging
import subprocess
from pathlib import Path
from typing import Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from optimum.intel import OVModelForCausalLM
from app.core.config import config


logger = logging.getLogger(__name__)


class ModelConverter:
    """Model converter class for OpenVINO format"""
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = Path(models_dir)
        
    def convert_to_openvino(
        self,
        source_model_path: Path,
        output_path: Path,
        precision: str = "FP16",
        quantize: bool = True
    ) -> Path:
        """
        Convert HuggingFace model to OpenVINO format
        
        Args:
            source_model_path: Path to source HuggingFace model
            output_path: Path to save OpenVINO model
            precision: Model precision (FP16, FP32)
            quantize: Whether to apply INT8 quantization
            
        Returns:
            Path to converted OpenVINO model
        """
        logger.info(f"Converting model from {source_model_path} to {output_path}")
        
        try:
            # Create output directory
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Load tokenizer
            logger.info("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(source_model_path)
            
            # Save tokenizer to output path
            tokenizer.save_pretrained(output_path)
            
            # Convert model using optimum-intel
            logger.info("Converting model to OpenVINO format...")
            
            if quantize:
                logger.info("Applying INT8 quantization...")
                ov_model = OVModelForCausalLM.from_pretrained(
                    source_model_path,
                    export=True,
                    load_in_8bit=True
                )
            else:
                ov_model = OVModelForCausalLM.from_pretrained(
                    source_model_path,
                    export=True
                )
            
            # Save converted model
            ov_model.save_pretrained(output_path)
            
            logger.info(f"Model converted successfully to {output_path}")
            return output_path
            
        except Exception as e:
            logger.error(f"Failed to convert model: {str(e)}")
            raise
    
    def check_openvino_model_exists(self, model_path: Path) -> bool:
        """Check if OpenVINO model files exist"""
        required_files = ["openvino_model.xml", "openvino_model.bin"]
        return all((model_path / file).exists() for file in required_files)
    
    def get_openvino_model_path(self, model_name: str) -> Path:
        """Get path to OpenVINO model"""
        return self.models_dir / f"{model_name}_ov"
    
    def convert_model_if_needed(self, model_name: str, source_path: Path) -> Path:
        """Convert model to OpenVINO format if not already converted"""
        ov_model_path = self.get_openvino_model_path(model_name)
        
        if self.check_openvino_model_exists(ov_model_path):
            logger.info(f"OpenVINO model already exists at {ov_model_path}")
            return ov_model_path
        
        logger.info(f"Converting model {model_name} to OpenVINO format...")
        return self.convert_to_openvino(
            source_path,
            ov_model_path,
            precision=config.get("hardware.precision", "FP16"),
            quantize=True
        )