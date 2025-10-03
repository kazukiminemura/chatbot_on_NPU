"""Chat API endpoints."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from ..models import model_manager
from ..core import logger

router = APIRouter()


class ChatRequest(BaseModel):
    """Chat request model."""
    message: str = Field(..., min_length=1, max_length=4000)
    max_tokens: Optional[int] = Field(None, ge=1, le=2000)
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    top_k: Optional[int] = Field(None, ge=1, le=100)
    repetition_penalty: Optional[float] = Field(None, ge=1.0, le=2.0)


class ChatResponse(BaseModel):
    """Chat response model."""
    response: str
    inference_time: float
    tokens_generated: int
    error: bool = False


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str
    model_loaded: bool
    npu_available: bool
    memory_usage: str
    details: Dict[str, Any]


@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        status_info = model_manager.get_status()
        
        return HealthResponse(
            status="healthy" if status_info.get("initialized") else "initializing",
            model_loaded=status_info.get("model_loaded", False),
            npu_available=status_info.get("device") == "NPU",
            memory_usage=f"{status_info.get('memory_usage_gb', 0):.1f}GB",
            details=status_info
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Chat endpoint for single response."""
    try:
        if not model_manager.is_initialized:
            raise HTTPException(status_code=503, detail="Model not initialized")
        
        # Generate response
        result = await model_manager.generate_single_response(
            prompt=request.message,
            max_tokens=request.max_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty
        )
        
        if result.get("error"):
            raise HTTPException(status_code=500, detail=result["response"])
        
        return ChatResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/model/status")
async def model_status():
    """Get detailed model status."""
    try:
        return model_manager.get_status()
    except Exception as e:
        logger.error(f"Model status error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/model/reload")
async def reload_model():
    """Reload the model."""
    try:
        success = await model_manager.reload_model()
        if success:
            return {"message": "Model reloaded successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to reload model")
    except Exception as e:
        logger.error(f"Model reload error: {e}")
        raise HTTPException(status_code=500, detail=str(e))