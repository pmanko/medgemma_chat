from typing import Any, Dict, Optional, List
from pydantic import BaseModel, Field


class A2AMessage(BaseModel):
    message_id: str
    correlation_id: str
    sender_id: str
    receiver_id: str
    task_name: str
    payload: Dict[str, Any]
    status: str
    error: Optional[Dict[str, Any]] = None


class ChatRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    conversation_id: Optional[str] = None
    scope: Optional[str] = Field(default="hie", description="facility | hie")
    facility_id: Optional[str] = None
    org_ids: Optional[List[str]] = None


class ChatResponse(BaseModel):
    response: str
    correlation_id: str 