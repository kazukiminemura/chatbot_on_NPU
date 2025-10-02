"""
Application runner for Llama2-7B NPU Chatbot
"""
import sys
import uvicorn
from pathlib import Path

# Add app directory to Python path
app_dir = Path(__file__).parent / "app"
sys.path.insert(0, str(app_dir.parent))

from app.core.config import config
from app.core.logger import setup_logging


def main():
    """Main entry point"""
    # Setup logging
    setup_logging(
        log_level=config.get("server.log_level", "INFO"),
        log_dir="logs"
    )
    
    # Get server configuration
    server_config = config.get_server_config()
    
    print("🤖 Starting Llama2-7B NPU Chatbot...")
    print(f"📡 Server: http://{server_config.get('host', 'localhost')}:{server_config.get('port', 8000)}")
    print("🔧 Setting up model (this may take a few minutes on first run)...")
    
    # Run the application
    uvicorn.run(
        "app.main:app",
        host=server_config.get("host", "localhost"),
        port=server_config.get("port", 8000),
        reload=False,
        log_level=server_config.get("log_level", "info").lower()
    )


if __name__ == "__main__":
    main()