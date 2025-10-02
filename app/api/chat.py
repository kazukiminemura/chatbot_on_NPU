"""
Chat API endpoints for Llama2-7B NPU Chatbot
"""
import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Optional
from app.models.model_manager import model_manager


logger = logging.getLogger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    message: str
    max_tokens: Optional[int] = None
    temperature: Optional[float] = None
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    repetition_penalty: Optional[float] = None


class ChatResponse(BaseModel):
    response: str
    inference_time: float
    tokens_generated: int


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Generate chat response
    
    Args:
        request: Chat request with message and parameters
        
    Returns:
        Generated response with metadata
    """
    try:
        if not model_manager.is_model_ready():
            raise HTTPException(status_code=503, detail="Model not ready")
        
        inference_engine = model_manager.get_inference_engine()
        
        result = inference_engine.generate_single_response(
            prompt=request.message,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty
        )
        
        return ChatResponse(**result)
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/info")
async def get_model_info():
    """Get model information and status"""
    try:
        return model_manager.get_model_info()
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))