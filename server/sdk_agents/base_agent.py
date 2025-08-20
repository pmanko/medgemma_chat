"""
Base A2A Agent implementation using FastAPI
Provides common functionality for all A2A agents
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from a2a.types import (
    AgentCard,
    SendMessageRequest,
    SendMessageSuccessResponse,
    Task,
    TaskStatus,
    TaskState,
    Message,
    TextPart,
    A2AError,
    MethodNotFoundError
)
import logging
from typing import Dict, Any, Optional
import uuid
from datetime import datetime
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class BaseA2AAgent(ABC):
    """Base class for A2A agents using FastAPI"""
    
    def __init__(self, name: str, description: str, version: str = "1.0.0"):
        self.name = name
        self.description = description
        self.version = version
        self.app = FastAPI(title=name)
        self._setup_routes()
    
    def _setup_routes(self):
        """Set up the A2A protocol routes"""
        
        @self.app.get("/.well-known/agent.json")
        async def get_agent_card():
            """Return the agent card"""
            card = await self.get_agent_card()
            return card.model_dump(exclude_none=True)
        
        @self.app.post("/")
        async def handle_jsonrpc(request: Request):
            """Handle JSON-RPC requests"""
            try:
                body = await request.json()
                
                # Parse the JSON-RPC request
                method = body.get("method")
                params = body.get("params", {})
                request_id = body.get("id")
                
                # Route to appropriate handler
                if method == "send_message":
                    result = await self.handle_send_message(params)
                elif method == "get_agent_card":
                    card = await self.get_agent_card()
                    result = card.model_dump(exclude_none=True)
                else:
                    # Method not found
                    error = MethodNotFoundError(
                        code=-32601,
                        message=f"Method {method} not found"
                    )
                    return JSONResponse(
                        content={
                            "jsonrpc": "2.0",
                            "error": error.model_dump(),
                            "id": request_id
                        },
                        status_code=200
                    )
                
                # Return success response
                return JSONResponse(
                    content={
                        "jsonrpc": "2.0",
                        "result": result,
                        "id": request_id
                    },
                    status_code=200
                )
                
            except Exception as e:
                logger.error(f"Error handling request: {e}", exc_info=True)
                error = A2AError(
                    code=-32603,
                    message=str(e)
                )
                return JSONResponse(
                    content={
                        "jsonrpc": "2.0",
                        "error": error.model_dump(),
                        "id": body.get("id") if 'body' in locals() else None
                    },
                    status_code=200
                )
    
    async def handle_send_message(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle send_message requests"""
        message = params.get("message", {})
        
        # Extract the user's query from the message parts
        query = ""
        for part in message.get("parts", []):
            if part.get("type") == "text":
                query = part.get("text", "")
                break
        
        # Process the message
        response_text = await self.process_message(query)
        
        # Create a task response
        task_id = str(uuid.uuid4())
        task = Task(
            id=task_id,
            status=TaskStatus(
                state=TaskState.completed,
                message="Task completed successfully",
                timestamp=datetime.utcnow().isoformat()
            ),
            history=[
                Message(
                    role="user",
                    parts=[TextPart(text=query)],
                    message_id=message.get("messageId", str(uuid.uuid4()))
                ),
                Message(
                    role="agent",
                    parts=[TextPart(text=response_text)],
                    message_id=str(uuid.uuid4())
                )
            ]
        )
        
        return task.model_dump(exclude_none=True)
    
    @abstractmethod
    async def get_agent_card(self) -> AgentCard:
        """Return the agent's capabilities card"""
        pass
    
    @abstractmethod
    async def process_message(self, query: str) -> str:
        """Process a message and return a response"""
        pass
    
    async def cleanup(self):
        """Clean up resources"""
        pass
