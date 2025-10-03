"""OpenVINO inference engine for DeepSeek-R1-Distill-Qwen-1.5B model."""
import time
import asyncio
from typing import AsyncGenerator, Optional, Dict, Any
import numpy as np
from pathlib import Path

try:
    import openvino as ov
    from optimum.intel.openvino import OVModelForCausalLM
    from transformers import AutoTokenizer, TextStreamer
    OPENVINO_AVAILABLE = True
except ImportError:
    OPENVINO_AVAILABLE = False

from ..core import config_manager, logger


class OpenVINOInferenceEngine:
    """OpenVINO inference engine for DeepSeek model."""
    
    def __init__(self):
        self.model: Optional[OVModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.core: Optional[ov.Core] = None
        self.config = config_manager.config
        self.model_path = config_manager.get_model_path()
        self.is_loaded = False
        
    async def initialize(self) -> bool:
        """Initialize the OpenVINO inference engine."""
        if not OPENVINO_AVAILABLE:
            logger.error("OpenVINO is not available. Please install openvino and optimum-intel.")
            return False
            
        try:
            logger.info("Initializing OpenVINO inference engine...")
            
            # Initialize OpenVINO core
            self.core = ov.Core()
            
            # Check NPU availability
            available_devices = self.core.available_devices
            logger.info(f"Available devices: {available_devices}")
            
            if "NPU" not in available_devices:
                logger.warning("NPU device not available. Falling back to CPU.")
                self.config.hardware.device = "CPU"
            
            # Load model and tokenizer
            await self._load_model()
            
            self.is_loaded = True
            logger.info("OpenVINO inference engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenVINO inference engine: {e}")
            return False
    
    async def _load_model(self):
        """Load the DeepSeek model and tokenizer."""
        try:
            logger.info(f"Loading model from {self.config.model.repo_id}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.config.model.repo_id,
                trust_remote_code=True
            )
            
            # Set padding token if not exists
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load OpenVINO model
            device = self.config.hardware.device
            compile_config = self.config.openvino.compile_config if device == "NPU" else {}
            
            self.model = OVModelForCausalLM.from_pretrained(
                self.config.model.repo_id,
                device=device,
                ov_config=compile_config,
                trust_remote_code=True,
                export=False  # Model is already in OpenVINO format
            )
            
            logger.info(f"Model loaded successfully on {device}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    async def generate_response(
        self, 
        prompt: str, 
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        top_k: Optional[int] = None,
        repetition_penalty: Optional[float] = None
    ) -> AsyncGenerator[str, None]:
        """Generate streaming response from the model."""
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call initialize() first.")
        
        try:
            # Use config defaults if parameters not provided
            max_tokens = max_tokens or self.config.inference.max_tokens
            temperature = temperature or self.config.inference.temperature
            top_p = top_p or self.config.inference.top_p
            top_k = top_k or self.config.inference.top_k
            repetition_penalty = repetition_penalty or self.config.inference.repetition_penalty
            
            # Format prompt for chat
            formatted_prompt = self._format_chat_prompt(prompt)
            
            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt", 
                truncation=True,
                max_length=self.config.model.max_context_length - max_tokens
            )
            
            # Generation parameters
            generation_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "do_sample": self.config.inference.do_sample,
                "pad_token_id": self.tokenizer.eos_token_id,
                "eos_token_id": self.tokenizer.eos_token_id,
            }
            
            # Generate with streaming
            start_time = time.time()
            generated_tokens = 0
            
            # Use model generate method with streaming
            with self.model.model.request() as request:
                # Set input
                input_ids = inputs["input_ids"]
                request.set_tensor("input_ids", input_ids.numpy())
                
                # Start inference
                request.start_async()
                request.wait()
                
                # Get initial output
                output = request.get_output_tensor()
                output_ids = output.data
                
                # Simple generation loop (this is a simplified version)
                # In practice, you'd implement proper streaming with the OpenVINO model
                generated_ids = self.model.generate(
                    **inputs,
                    **generation_kwargs,
                    streamer=None  # We'll implement custom streaming below
                )
                
                # Decode and yield tokens
                new_tokens = generated_ids[0][inputs["input_ids"].shape[1]:]
                
                current_text = ""
                for i, token_id in enumerate(new_tokens):
                    token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                    if token_text:
                        current_text += token_text
                        generated_tokens += 1
                        yield token_text
                        
                        # Add small delay to simulate streaming
                        await asyncio.sleep(0.01)
                        
                        # Check for stop conditions
                        if token_id == self.tokenizer.eos_token_id:
                            break
            
            # Log performance metrics
            total_time = time.time() - start_time
            tokens_per_second = generated_tokens / total_time if total_time > 0 else 0
            logger.info(f"Generated {generated_tokens} tokens in {total_time:.2f}s ({tokens_per_second:.2f} tokens/s)")
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            yield f"Error: {str(e)}"
    
    def _format_chat_prompt(self, user_message: str) -> str:
        """Format the user message for the DeepSeek model."""
        # DeepSeek-R1 models typically use a specific chat format
        # This is a simplified version - adjust based on the model's requirements
        return f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{user_message}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    
    async def generate_single_response(
        self, 
        prompt: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """Generate a single non-streaming response."""
        start_time = time.time()
        full_response = ""
        token_count = 0
        
        async for token in self.generate_response(prompt, **kwargs):
            if not token.startswith("Error:"):
                full_response += token
                token_count += 1
            else:
                return {
                    "response": token,
                    "inference_time": time.time() - start_time,
                    "tokens_generated": 0,
                    "error": True
                }
        
        return {
            "response": full_response.strip(),
            "inference_time": time.time() - start_time,
            "tokens_generated": token_count,
            "error": False
        }
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model."""
        if not self.is_loaded:
            return {"loaded": False}
        
        return {
            "loaded": True,
            "model_name": self.config.model.name,
            "model_type": self.config.model.model_type,
            "device": self.config.hardware.device,
            "precision": self.config.model.precision,
            "max_context_length": self.config.model.max_context_length,
            "vocab_size": len(self.tokenizer) if self.tokenizer else 0
        }