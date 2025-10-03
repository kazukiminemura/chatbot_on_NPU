"""WebSocket handler for real-time chat."""
import json
import asyncio
from typing import Dict, Any
from fastapi import WebSocket, WebSocketDisconnect
from ..models import model_manager
from ..core import logger


class WebSocketManager:
    """Manages WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Accept a WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"WebSocket client {client_id} connected")
    
    def disconnect(self, client_id: str):
        """Remove a WebSocket connection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
            logger.info(f"WebSocket client {client_id} disconnected")
    
    async def send_message(self, client_id: str, message: Dict[str, Any]):
        """Send a message to a specific client."""
        if client_id in self.active_connections:
            try:
                await self.active_connections[client_id].send_text(json.dumps(message))
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {e}")
                self.disconnect(client_id)


# Global WebSocket manager
websocket_manager = WebSocketManager()


async def handle_websocket_chat(websocket: WebSocket, client_id: str):
    """Handle WebSocket chat communication."""
    await websocket_manager.connect(websocket, client_id)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            message_data = json.loads(data)
            
            if message_data.get("type") == "message":
                await process_chat_message(client_id, message_data.get("data", {}))
            elif message_data.get("type") == "ping":
                await websocket_manager.send_message(client_id, {"type": "pong"})
            
    except WebSocketDisconnect:
        websocket_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {e}")
        await websocket_manager.send_message(client_id, {
            "type": "error",
            "data": {"message": str(e)}
        })
        websocket_manager.disconnect(client_id)


async def process_chat_message(client_id: str, message_data: Dict[str, Any]):
    """Process a chat message and send streaming response."""
    try:
        if not model_manager.is_initialized:
            await websocket_manager.send_message(client_id, {
                "type": "error",
                "data": {"message": "Model not initialized"}
            })
            return
        
        prompt = message_data.get("message", "")
        settings = message_data.get("settings", {})
        
        if not prompt:
            await websocket_manager.send_message(client_id, {
                "type": "error",
                "data": {"message": "Empty message"}
            })
            return
        
        # Send start signal
        await websocket_manager.send_message(client_id, {
            "type": "start",
            "data": {}
        })
        
        # Generate streaming response
        token_count = 0
        start_time = asyncio.get_event_loop().time()
        
        try:
            async for token in model_manager.generate_response(prompt, **settings):
                if token.startswith("Error:"):
                    await websocket_manager.send_message(client_id, {
                        "type": "error", 
                        "data": {"message": token}
                    })
                    return
                
                # Send token
                await websocket_manager.send_message(client_id, {
                    "type": "token",
                    "data": {
                        "token": token,
                        "is_final": False
                    }
                })
                token_count += 1
                
                # Small delay to prevent overwhelming the client
                await asyncio.sleep(0.01)
        
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            await websocket_manager.send_message(client_id, {
                "type": "error",
                "data": {"message": f"Generation error: {str(e)}"}
            })
            return
        
        # Send completion signal
        end_time = asyncio.get_event_loop().time()
        await websocket_manager.send_message(client_id, {
            "type": "complete",
            "data": {
                "inference_time": end_time - start_time,
                "total_tokens": token_count
            }
        })
        
    except Exception as e:
        logger.error(f"Error processing chat message: {e}")
        await websocket_manager.send_message(client_id, {
            "type": "error",
            "data": {"message": f"Processing error: {str(e)}"}
        })