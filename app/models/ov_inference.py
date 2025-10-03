"""OpenVINO GenAI inference engine for DeepSeek-R1-Distill-Qwen-1.5B model."""
import time
import asyncio
from typing import AsyncGenerator, Optional, Dict, Any
import numpy as np
from pathlib import Path

# Try to import OpenVINO GenAI components, fall back gracefully if not available
OPENVINO_GENAI_AVAILABLE = False
try:
    import openvino_genai as ov_genai
    OPENVINO_GENAI_AVAILABLE = True
except ImportError as e:
    print(f"Warning: OpenVINO GenAI not available - {e}")
    print("Running in simulation mode. Install OpenVINO GenAI for actual inference.")
    print("pip install openvino-genai")

from ..core import config_manager, logger
from ..utils.download import download_model, check_model_exists


class OpenVINOInferenceEngine:
    """OpenVINO GenAI inference engine for DeepSeek model."""
    
    def __init__(self):
        self.llm_pipe: Optional[ov_genai.LLMPipeline] = None
        self.config = config_manager.config
        self.model_path = config_manager.get_model_path()
        self.is_loaded = False
        self.device = "CPU"  # Default device
        
    async def initialize(self) -> bool:
        """Initialize the OpenVINO GenAI inference engine."""
        if not OPENVINO_GENAI_AVAILABLE:
            logger.warning("OpenVINO GenAI is not available. Running in simulation mode.")
            logger.info("To use actual model inference, install OpenVINO GenAI:")
            logger.info("pip install openvino-genai")
            # Simulate successful initialization for testing
            self.is_loaded = True
            return True
            
        try:
            logger.info("Initializing OpenVINO GenAI inference engine...")
            
            # Determine device to use
            self.device = self.config.hardware.device
            logger.info(f"Target device: {self.device}")
            
            # Load model using OpenVINO GenAI
            await self._load_model()
            
            self.is_loaded = True
            logger.info("OpenVINO GenAI inference engine initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenVINO GenAI inference engine: {e}")
            return False
    
    async def _load_model(self):
        """Load the DeepSeek model using OpenVINO GenAI."""
        try:
            logger.info(f"Loading model from {self.config.model.repo_id}")
            
            # Prepare model path and check if it exists locally
            model_repo = self.config.model.repo_id
            local_model_path = Path("./models") / model_repo.replace("/", "_")
            
            logger.info(f"Checking local model path: {local_model_path}")
            
            # Check if model exists locally, if not download it
            if not self._check_local_model(local_model_path):
                logger.info("Model not found locally. Starting download...")
                await self._download_model_if_needed(model_repo, local_model_path)
            else:
                logger.info("Model found locally. Using cached version.")
            
            # Use local path if it exists, otherwise use repo_id for direct download
            model_path_to_use = str(local_model_path) if local_model_path.exists() else model_repo
            logger.info(f"Using model path: {model_path_to_use}")
            
            # Create generation config
            generation_config = ov_genai.GenerationConfig()
            generation_config.max_new_tokens = self.config.inference.max_tokens
            generation_config.temperature = self.config.inference.temperature
            generation_config.top_p = self.config.inference.top_p
            generation_config.top_k = self.config.inference.top_k
            generation_config.repetition_penalty = self.config.inference.repetition_penalty
            generation_config.do_sample = self.config.inference.do_sample
            
            # Prepare device-specific configuration
            device_config = {"CACHE_DIR": str(Path("./cache").absolute())}
            
            if self.device == "NPU":
                # NPU-specific configuration
                device_config.update({
                    "NPU_USE_NPUW": "YES",
                    "PERFORMANCE_HINT": "LATENCY",
                    "CACHE_MODE": "OPTIMIZE_SPEED"
                })
                logger.info("Using NPU-specific configuration")
            elif self.device == "CPU":
                # CPU-specific configuration
                device_config.update({
                    "PERFORMANCE_HINT": "LATENCY",
                    "INFERENCE_NUM_THREADS": "4"
                })
                logger.info("Using CPU-specific configuration")
            
            logger.info(f"Device config: {device_config}")
            
            # Try to load the model with the specified device
            try:
                logger.info(f"Loading model on {self.device}")
                
                # Load LLM pipeline with OpenVINO GenAI
                # Use the correct parameter name: models_path (not model_path)
                # The first argument is the models_path (positional argument)
                self.llm_pipe = ov_genai.LLMPipeline(
                    model_path_to_use,  # models_path as first positional argument
                    device=self.device,
                    config=device_config
                )
                
                logger.info(f"Model loaded successfully on {self.device}")
                
            except Exception as model_error:
                logger.warning(f"Failed to load on {self.device}: {model_error}")
                logger.debug(f"Model error details: {type(model_error).__name__}: {str(model_error)}")
                
                if self.device == "NPU":
                    logger.info("Trying to load on CPU as fallback...")
                    
                    try:
                        # Fallback to CPU
                        self.device = "CPU"
                        cpu_config = {"CACHE_DIR": str(Path("./cache").absolute())}
                        self.llm_pipe = ov_genai.LLMPipeline(
                            model_path_to_use,  # models_path as first positional argument
                            device="CPU",
                            config=cpu_config
                        )
                        logger.info("Model loaded successfully on CPU (fallback)")
                        self.config.hardware.device = "CPU"
                    except Exception as cpu_error:
                        logger.error(f"Failed to load on CPU fallback: {cpu_error}")
                        raise cpu_error
                else:
                    raise model_error
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def _check_local_model(self, model_path: Path) -> bool:
        """Check if the model exists locally with required files."""
        if not model_path.exists():
            return False
        
        # Check for essential OpenVINO GenAI model files
        required_files = [
            "openvino_model.xml",
            "openvino_model.bin",
            "config.json"
        ]
        
        # Also check for alternative file patterns
        alternative_patterns = [
            "*.xml",
            "*.bin"
        ]
        
        # First check for exact files
        for file_name in required_files:
            if (model_path / file_name).exists():
                logger.debug(f"Found required file: {file_name}")
                continue
        
        # If exact files not found, check for pattern matches
        xml_files = list(model_path.glob("*.xml"))
        bin_files = list(model_path.glob("*.bin"))
        
        if xml_files and bin_files:
            logger.info(f"Found OpenVINO model files: {len(xml_files)} XML, {len(bin_files)} BIN")
            return True
        
        logger.warning(f"Model directory exists but missing required files at {model_path}")
        return False
    
    async def _download_model_if_needed(self, repo_id: str, local_path: Path) -> None:
        """Download the model if it doesn't exist locally."""
        try:
            logger.info(f"Downloading model {repo_id} to {local_path}")
            
            # Create directory if it doesn't exist
            local_path.mkdir(parents=True, exist_ok=True)
            
            # Download using the download utility
            downloaded_path = download_model(
                repo_id=repo_id,
                local_dir=str(local_path)
            )
            
            logger.info(f"Model downloaded successfully to {downloaded_path}")
            
        except Exception as e:
            logger.error(f"Failed to download model {repo_id}: {e}")
            logger.info("Will attempt to use direct repo_id for loading")
            # Don't raise the error, let the main loading try with repo_id directly
    
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
        
        # If OpenVINO GenAI is not available, run simulation
        if not OPENVINO_GENAI_AVAILABLE:
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
            
            # Create generation config for this request
            generation_config = ov_genai.GenerationConfig()
            generation_config.max_new_tokens = max_tokens
            generation_config.temperature = temperature
            generation_config.top_p = top_p
            generation_config.top_k = top_k
            generation_config.repetition_penalty = repetition_penalty
            generation_config.do_sample = self.config.inference.do_sample
            
            # Generate with streaming using OpenVINO GenAI
            start_time = time.time()
            generated_tokens = 0
            
            # Create a simple callback streamer for token streaming
            class TokenStreamer:
                def __init__(self):
                    self.tokens = []
                
                def put(self, token):
                    self.tokens.append(token)
                
                def end(self):
                    pass
            
            # For now, use non-streaming generation and simulate streaming
            # This can be improved when proper streaming API is available
            full_response = self.llm_pipe.generate(formatted_prompt, generation_config)
            
            # Simulate streaming by yielding words
            words = full_response.split()
            for word in words:
                generated_tokens += 1
                yield word + " "
                await asyncio.sleep(0.01)  # Small delay for streaming effect
            
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
            "device": self.device,
            "precision": self.config.model.precision,
            "max_context_length": self.config.model.max_context_length,
            "inference_engine": "OpenVINO GenAI"
        }
        
        if OPENVINO_GENAI_AVAILABLE and self.llm_pipe:
            # Try to get additional model info from the pipeline
            try:
                # Note: API may vary depending on OpenVINO GenAI version
                base_info["vocab_size"] = "Available via OpenVINO GenAI"
            except:
                base_info["vocab_size"] = "N/A"
        else:
            base_info["vocab_size"] = "N/A (Simulation Mode)"
            base_info["simulation_mode"] = True
        
        return base_info