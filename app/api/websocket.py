"""
WebSocket endpoints for real-time chat
"""
import json
import logging
import asyncio
from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Any
from app.models.model_manager import model_manager


logger = logging.getLogger(__name__)


class WebSocketManager:
    """WebSocket connection manager"""
    
    def __init__(self):
        self.active_connections: list[WebSocket] = []
    
    async def connect(self, websocket: WebSocket):
        """Accept WebSocket connection"""
        await websocket.accept()
        self.active_connections.append(websocket)
        logger.info(f"WebSocket connected. Total connections: {len(self.active_connections)}")
    
    def disconnect(self, websocket: WebSocket):
        """Remove WebSocket connection"""
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)
        logger.info(f"WebSocket disconnected. Total connections: {len(self.active_connections)}")
    
    async def send_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Send message to WebSocket"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            logger.error(f"Error sending WebSocket message: {str(e)}")


# Global WebSocket manager
websocket_manager = WebSocketManager()


async def websocket_chat_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time chat
    """
    await websocket_manager.connect(websocket)
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                await handle_websocket_message(websocket, message_data)
            except json.JSONDecodeError:
                await websocket_manager.send_message(websocket, {
                    "type": "error",
                    "data": {"error": "Invalid JSON format"}
                })
            except Exception as e:
                logger.error(f"Error handling WebSocket message: {str(e)}")
                await websocket_manager.send_message(websocket, {
                    "type": "error",
                    "data": {"error": str(e)}
                })
                
    except WebSocketDisconnect:
        websocket_manager.disconnect(websocket)


async def handle_websocket_message(websocket: WebSocket, message_data: Dict[str, Any]):
    """
    Handle incoming WebSocket message
    
    Args:
        websocket: WebSocket connection
        message_data: Parsed message data
    """
    message_type = message_data.get("type")
    
    if message_type == "message":
        await handle_chat_message(websocket, message_data.get("data", {}))
    elif message_type == "ping":
        await websocket_manager.send_message(websocket, {"type": "pong", "data": {}})
    else:
        await websocket_manager.send_message(websocket, {
            "type": "error",
            "data": {"error": f"Unknown message type: {message_type}"}
        })


async def handle_chat_message(websocket: WebSocket, data: Dict[str, Any]):
    """
    Handle chat message and generate streaming response
    
    Args:
        websocket: WebSocket connection
        data: Chat message data
    """
    if not model_manager.is_model_ready():
        await websocket_manager.send_message(websocket, {
            "type": "error",
            "data": {"error": "Model not ready"}
        })
        return
    
    message = data.get("message", "")
    settings = data.get("settings", {})
    
    if not message.strip():
        await websocket_manager.send_message(websocket, {
            "type": "error",
            "data": {"error": "Empty message"}
        })
        return
    
    try:
        inference_engine = model_manager.get_inference_engine()
        
        # Send start signal
        await websocket_manager.send_message(websocket, {
            "type": "start",
            "data": {}
        })
        
        # Generate streaming response
        token_count = 0
        start_time = asyncio.get_event_loop().time()
        
        for token in inference_engine.generate_response(
            prompt=message,
            max_tokens=settings.get("max_tokens"),
            temperature=settings.get("temperature"),
            top_p=settings.get("top_p"),
            top_k=settings.get("top_k"),
            repetition_penalty=settings.get("repetition_penalty")
        ):
            # Send token
            await websocket_manager.send_message(websocket, {
                "type": "token",
                "data": {
                    "token": token,
                    "is_final": False
                }
            })
            
            token_count += 1
            
            # Small delay to allow for smooth streaming
            await asyncio.sleep(0.01)
        
        # Send completion signal
        end_time = asyncio.get_event_loop().time()
        inference_time = end_time - start_time
        
        await websocket_manager.send_message(websocket, {
            "type": "complete",
            "data": {
                "inference_time": inference_time,
                "total_tokens": token_count
            }
        })
        
    except Exception as e:
        logger.error(f"Error in streaming response: {str(e)}")
        await websocket_manager.send_message(websocket, {
            "type": "error",
            "data": {"error": str(e)}
        })