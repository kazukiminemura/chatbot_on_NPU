"""
Main FastAPI application for Llama2-7B NPU Chatbot
"""
import logging
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pathlib import Path

from app.core.config import config
from app.core.logger import setup_logging
from app.models.model_manager import model_manager
from app.api.chat import router as chat_router
from app.api.websocket import websocket_chat_endpoint


logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    # Startup
    logger.info("Starting Llama2-7B NPU Chatbot application...")
    
    try:
        # Setup model in background
        logger.info("Setting up model...")
        await asyncio.create_task(asyncio.to_thread(model_manager.setup_model))
        logger.info("Model setup completed")
    except Exception as e:
        logger.error(f"Failed to setup model: {str(e)}")
        # Continue without model - will show error in UI
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")
    model_manager.cleanup()


# Create FastAPI application
app = FastAPI(
    title="Llama2-7B NPU Chatbot",
    description="AI Chatbot powered by Llama2-7B running on Intel NPU",
    version="1.0.0",
    lifespan=lifespan
)

# Include API routers
app.include_router(chat_router, prefix="/api")

# Mount static files
static_path = Path("static")
if static_path.exists():
    app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main chat interface"""
    try:
        with open("static/index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    except FileNotFoundError:
        return HTMLResponse(content="""
        <html>
            <head><title>Llama2-7B NPU Chatbot</title></head>
            <body>
                <h1>Llama2-7B NPU Chatbot</h1>
                <p>Static files not found. Please ensure the static directory exists.</p>
            </body>
        </html>
        """)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        model_info = model_manager.get_model_info()
        return {
            "status": "healthy",
            "model_loaded": model_info.get("is_loaded", False),
            "npu_available": True,  # TODO: Add actual NPU detection
            "model_info": model_info
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "model_loaded": False,
            "npu_available": False
        }


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for real-time chat"""
    await websocket_chat_endpoint(websocket)


@app.get("/api/status")
async def get_status():
    """Get application status"""
    return {
        "application": "Llama2-7B NPU Chatbot",
        "version": "1.0.0",
        "model_ready": model_manager.is_model_ready(),
        "config": {
            "model": config.get_model_config(),
            "server": config.get_server_config(),
            "hardware": config.get_hardware_config()
        }
    }


if __name__ == "__main__":
    import uvicorn
    
    # Setup logging
    setup_logging(
        log_level=config.get("server.log_level", "INFO"),
        log_dir="logs"
    )
    
    server_config = config.get_server_config()
    
    uvicorn.run(
        "app.main:app",
        host=server_config.get("host", "localhost"),
        port=server_config.get("port", 8000),
        reload=False,
        log_level=server_config.get("log_level", "info").lower()
    )