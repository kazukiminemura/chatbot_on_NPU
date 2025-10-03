"""Main FastAPI application."""
import asyncio
import uvicorn
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os
import sys

# Add error handling for imports
try:
    from .core import config_manager, logger
    from .models import model_manager
    from .api import chat_router, handle_websocket_chat
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure all dependencies are installed:")
    print("pip install -r requirements.txt")
    sys.exit(1)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting DeepSeek-R1-Distill-Qwen-1.5B NPU Chatbot...")
    
    # Initialize model manager
    success = await model_manager.initialize()
    if not success:
        logger.error("Failed to initialize model manager")
        raise RuntimeError("Model initialization failed")
    
    logger.info("Application startup complete")
    yield
    
    # Shutdown
    logger.info("Shutting down application...")


# Create FastAPI app
app = FastAPI(
    title="DeepSeek-R1-Distill-Qwen-1.5B NPU Chatbot",
    description="AI chatbot powered by DeepSeek-R1-Distill-Qwen-1.5B running on Intel NPU",
    version="1.0.0",
    lifespan=lifespan
)

# Get configuration
config = config_manager.config

# Add CORS middleware
from fastapi.middleware.cors import CORSMiddleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(chat_router, prefix="/api")

# Static files
static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# WebSocket endpoint
@app.websocket("/ws/chat/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for real-time chat."""
    await handle_websocket_chat(websocket, client_id)

# Root endpoint
@app.get("/")
async def root():
    """Serve the main page."""
    static_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
    index_path = os.path.join(static_dir, "index.html")
    
    if os.path.exists(index_path):
        return FileResponse(index_path)
    else:
        return {
            "message": "DeepSeek-R1-Distill-Qwen-1.5B NPU Chatbot API",
            "docs": "/docs",
            "health": "/api/health"
        }


def main():
    """Run the application."""
    uvicorn.run(
        "app.main:app",
        host=config.server.host,
        port=config.server.port,
        reload=False,
        log_level=config.server.log_level.lower()
    )


if __name__ == "__main__":
    main()