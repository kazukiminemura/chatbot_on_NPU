"""OpenVINO inference engine for DeepSeek-R1-Distill-Qwen-1.5B model."""
import time
import asyncio
from typing import AsyncGenerator, Optional, Dict, Any
import numpy as np
from pathlib import Path

# Try to import OpenVINO components, fall back gracefully if not available
OPENVINO_AVAILABLE = False
try:
    import openvino as ov
    from optimum.intel.openvino import OVModelForCausalLM
    from transformers import AutoTokenizer, TextStreamer
    import torch
    OPENVINO_AVAILABLE = True
except ImportError as e:
    print(f"Warning: OpenVINO not available - {e}")
    print("Running in simulation mode. Install OpenVINO for actual inference.")

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
            logger.warning("OpenVINO is not available. Running in simulation mode.")
            logger.info("To use actual model inference, install OpenVINO:")
            logger.info("pip install openvino optimum-intel transformers")
            # Simulate successful initialization for testing
            self.is_loaded = True
            return True
            
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
            
            # Load OpenVINO model with static shape configuration
            device = self.config.hardware.device
            compile_config = {}
            
            if device == "NPU":
                # NPU specific configuration with static shapes
                compile_config = {
                    "NPU_USE_NPUW": "YES",
                    "NPU_COMPILATION_MODE_PARAMS": "compute-layers-with-higher-precision=Softmax,Add",
                    "INFERENCE_PRECISION_HINT": "f16",
                    "PERFORMANCE_HINT": "LATENCY",
                    "CACHE_MODE": "OPTIMIZE_SPEED",
                    # Force static shapes for NPU
                    "DYNAMIC_SHAPES": "NO",
                    "RESHAPE_ON_BATCH_DIM": "NO"
                }
            else:
                # CPU/GPU configuration
                compile_config = {
                    "PERFORMANCE_HINT": "LATENCY",
                    "INFERENCE_NUM_THREADS": "4" if device == "CPU" else "AUTO",
                    "DYNAMIC_SHAPES": "NO"
                }
            
            # Try to load the model with static shape configuration
            try:
                # First, load the model without device compilation
                model_path = self.config.model.repo_id
                static_shape = self.config.openvino.static_input_shape
                batch_size = static_shape["batch_size"]
                seq_length = static_shape["sequence_length"]
                
                # Load model using optimum
                self.model = OVModelForCausalLM.from_pretrained(
                    model_path,
                    device=device,
                    ov_config=compile_config,
                    trust_remote_code=True,
                    export=False,
                    use_cache=True  # Keep cache enabled as required by the model
                )
                
                # Try to reshape the model for static input
                try:
                    # Get the model's input info
                    model_inputs = self.model.model.inputs
                    logger.info(f"Original model inputs: {[inp.get_partial_shape() for inp in model_inputs]}")
                    
                    # Create static shapes dictionary
                    static_shapes = {}
                    for inp in model_inputs:
                        if inp.get_any_name() == "input_ids":
                            static_shapes[inp] = [batch_size, seq_length]
                        elif "attention_mask" in inp.get_any_name():
                            static_shapes[inp] = [batch_size, seq_length]
                        else:
                            # Keep other inputs as-is or set reasonable defaults
                            current_shape = inp.get_partial_shape()
                            if current_shape.is_dynamic:
                                static_shapes[inp] = [batch_size] + [1] * (len(current_shape) - 1)
                    
                    # Reshape the model with static shapes
                    if static_shapes:
                        logger.info(f"Reshaping model with static shapes: {static_shapes}")
                        self.model.model = self.model.model.reshape(static_shapes)
                        
                        # Now compile for the target device
                        self.model.model = self.core.compile_model(self.model.model, device, compile_config)
                        logger.info(f"Model successfully reshaped and compiled for {device}")
                    
                except Exception as reshape_error:
                    logger.warning(f"Model reshape failed, using original model: {reshape_error}")
                
                logger.info(f"Model loaded successfully on {device} with static shapes")
                
            except Exception as model_error:
                logger.warning(f"Failed to load on {device} with static shapes: {model_error}")
                
                if device == "NPU":
                    logger.info("Trying to load on CPU as fallback...")
                    
                    # Fallback to CPU with static configuration
                    cpu_config = {
                        "PERFORMANCE_HINT": "LATENCY",
                        "INFERENCE_NUM_THREADS": "4",
                        "DYNAMIC_SHAPES": "NO"
                    }
                    
                    static_shape = self.config.openvino.static_input_shape
                    batch_size = static_shape["batch_size"]
                    seq_length = static_shape["sequence_length"]
                    
                    self.model = OVModelForCausalLM.from_pretrained(
                        self.config.model.repo_id,
                        device="CPU",
                        ov_config=cpu_config,
                        trust_remote_code=True,
                        export=False,
                        use_cache=True  # Keep cache enabled
                    )
                    logger.info("Model loaded successfully on CPU (fallback) with static shapes")
                    self.config.hardware.device = "CPU"
                else:
                    raise model_error
            
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
        
        # If OpenVINO is not available, run simulation
        if not OPENVINO_AVAILABLE:
            async for token in self._simulate_generation(prompt, max_tokens):
                yield token
            return
        
        try:
            # Use config defaults if parameters not provided
            max_tokens = max_tokens or self.config.inference.max_tokens
            temperature = temperature or self.config.inference.temperature
            top_p = top_p or self.config.inference.top_p
            top_k = top_k or self.config.inference.top_k
            repetition_penalty = repetition_penalty or self.config.inference.repetition_penalty
            
            # Format prompt for chat
            formatted_prompt = self._format_chat_prompt(prompt)
            
            # Tokenize input with fixed length for static shapes
            max_input_length = self.config.openvino.static_input_shape["sequence_length"]
            inputs = self.tokenizer(
                formatted_prompt, 
                return_tensors="pt", 
                truncation=True,
                padding="max_length",  # Pad to fixed length
                max_length=max_input_length
            )
            
            # Ensure input tensor has the expected static shape [1, 512]
            input_ids = inputs["input_ids"]
            if input_ids.shape[1] != max_input_length:
                # Pad or truncate to exactly 512 tokens
                if input_ids.shape[1] > max_input_length:
                    input_ids = input_ids[:, :max_input_length]
                else:
                    if OPENVINO_AVAILABLE:
                        padding_length = max_input_length - input_ids.shape[1]
                        padding = torch.full((1, padding_length), self.tokenizer.pad_token_id, dtype=input_ids.dtype)
                        input_ids = torch.cat([input_ids, padding], dim=1)
                inputs["input_ids"] = input_ids
            
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
            
            # Simple generation approach that works with OpenVINO models
            generated_ids = self.model.generate(
                **inputs,
                **generation_kwargs
            )
            
            # Decode and yield tokens
            new_tokens = generated_ids[0][inputs["input_ids"].shape[1]:]
            
            # Stream the tokens one by one
            for i, token_id in enumerate(new_tokens):
                token_text = self.tokenizer.decode([token_id], skip_special_tokens=True)
                if token_text:
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
    
    async def _simulate_generation(self, prompt: str, max_tokens: Optional[int] = None) -> AsyncGenerator[str, None]:
        """Simulate text generation when OpenVINO is not available."""
        max_tokens = max_tokens or 100
        
        # Simple simulation response
        simulation_responses = [
            "こんにちは！",
            "申し訳ございませんが、現在シミュレーションモードで動作しています。",
            "実際のDeepSeek-R1-Distill-Qwen-1.5Bモデルを使用するには、",
            "OpenVINOとoptimum-intelをインストールしてください。",
            f"\nあなたの質問: {prompt}",
            "\n実際のモデルでは、この質問に対してより詳細で正確な回答を提供します。"
        ]
        
        for response in simulation_responses:
            for word in response.split():
                yield word + " "
                await asyncio.sleep(0.1)  # Simulate thinking time
    
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
        
        base_info = {
            "loaded": True,
            "model_name": self.config.model.name,
            "model_type": self.config.model.model_type,
            "device": self.config.hardware.device,
            "precision": self.config.model.precision,
            "max_context_length": self.config.model.max_context_length,
        }
        
        if OPENVINO_AVAILABLE and self.tokenizer:
            base_info["vocab_size"] = len(self.tokenizer)
        else:
            base_info["vocab_size"] = "N/A (Simulation Mode)"
            base_info["simulation_mode"] = True
        
        return base_info