import logging
import time
import psutil
import uuid
from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import agent_config, llm_config, a2a_endpoints
from .llm_clients import llm_client
from .a2a_layer import bus, registry, new_message
from .schemas import ChatRequest, ChatResponse
from .agents.user_proxy_agent import run_user_proxy_agent
from .agents.medgemma_agent import run_medgemma_agent
from .agents.fhir_agent import run_clinical_research_agent

import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

server_start_time = time.time()

app = FastAPI(
    title="A2A-Enabled Multi-Agent Medical Chat API",
    description="Direct chat endpoints + A2A-simulated /chat with agent threads",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatMessage(BaseModel):
    role: str = Field(...)
    content: str = Field(...)
    timestamp: Optional[str] = Field(default=None)


class PromptRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=4000)
    system_prompt: str = Field(default="", max_length=2000)
    max_new_tokens: int = Field(default=512, ge=1, le=2048)
    conversation_history: List[ChatMessage] = Field(default=[])
    conversation_id: Optional[str] = None


class PromptResponse(BaseModel):
    response: str


@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        if agent_config.enable_a2a:
            logger.info("Starting A2A simulation layer and agents...")
            bus.register_mailbox("web_ui")
            run_medgemma_agent()
            run_clinical_research_agent()
            run_user_proxy_agent()
            logger.info("Agents started")
        else:
            logger.info("A2A layer disabled by config")
    except Exception as e:
        logger.error(f"Failed to start agents: {e}")
        raise

    yield


app.router.lifespan_context = lifespan


@app.get("/")
def read_root():
    return {
        "status": "Server is running",
        "uptime_seconds": round(time.time() - server_start_time, 2),
        "a2a_enabled": agent_config.enable_a2a,
        "direct_models": {"general": llm_config.general_model, "medical": llm_config.med_model},
    }


@app.get("/manifest")
def get_manifest():
    return {"agents": registry.list_agents()}


@app.get("/health")
def health_check():
    uptime = time.time() - server_start_time
    memory_info = {}
    try:
        process = psutil.Process()
        memory_info["process_memory_gb"] = round(process.memory_info().rss / 1024**3, 2)
        memory_info["process_memory_percent"] = round(process.memory_percent(), 1)
    except Exception:
        pass
    return {"status": "healthy", "uptime_seconds": round(uptime, 2), "memory": memory_info, "timestamp": time.time()}


@app.post("/generate/general", response_model=PromptResponse)
def generate_general(request: PromptRequest):
    try:
        messages = []
        if request.system_prompt:
            messages.append({"role": "system", "content": request.system_prompt})
        for m in request.conversation_history[-16:]:
            messages.append({"role": m.role if m.role in ("user", "assistant", "system") else "user", "content": m.content})
        messages.append({"role": "user", "content": request.prompt})
        text = llm_client.generate_chat(
            model=llm_config.general_model,
            messages=messages,
            temperature=llm_config.temperature,
            max_tokens=min(request.max_new_tokens, 1024),
        )
        return {"response": text}
    except Exception as e:
        logger.error(f"General generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate/medgemma", response_model=PromptResponse)
def generate_medgemma(request: PromptRequest):
    try:
        system = request.system_prompt or (
            "You are a medical AI assistant. Provide accurate, evidence-based information. Include a brief disclaimer."
        )
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        for m in request.conversation_history[-16:]:
            messages.append({"role": m.role if m.role in ("user", "assistant", "system") else "user", "content": m.content})
        messages.append({"role": "user", "content": request.prompt})
        text = llm_client.generate_chat(
            model=llm_config.med_model,
            messages=messages,
            temperature=0.1,
            max_tokens=min(request.max_new_tokens, 800),
        )
        return {"response": text}
    except Exception as e:
        logger.error(f"Medical generation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not agent_config.enable_a2a:
        raise HTTPException(status_code=503, detail="A2A layer disabled")

    if agent_config.enable_a2a_native:
        if not a2a_endpoints.router_url:
            raise HTTPException(status_code=500, detail="A2A router URL is not configured")
        try:
            # Native A2A: forward to router's A2A endpoint
            # Expect router to expose an A2A-compatible HTTP endpoint accepting message-like payload
            resp = requests.post(
                a2a_endpoints.router_url.rstrip("/") + "/",
                json={
                    "sender_id": "web_ui",
                    "payload": {
                        "query": request.prompt,
                        "conversation_id": request.conversation_id,
                        "scope": request.scope,
                        "facility_id": request.facility_id,
                        "org_ids": request.org_ids,
                    },
                },
                timeout=agent_config.chat_timeout_seconds,
            )
            resp.raise_for_status()
            data = resp.json()
            # Normalize expected response structure
            text = (
                data.get("result", {}).get("content", {}).get("text")
                or data.get("result", {}).get("parts", [{}])[0].get("text")
                or data.get("result", {}).get("text")
                or data.get("result", {}).get("message", "")
                or data.get("response", "")
                or "(No content)"
            )
            return ChatResponse(response=text, correlation_id="native")
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    # Simulated A2A path below
    correlation_id = str(uuid.uuid4())
    try:
        bus.register_mailbox("web_ui")
        msg = new_message(
            sender_id="web_ui",
            receiver_id="user_proxy_agent",
            task_name="initiate_query",
            payload={
                "query": request.prompt,
                "conversation_id": request.conversation_id,
                "scope": request.scope,
                "facility_id": request.facility_id,
                "org_ids": request.org_ids,
            },
            correlation_id=correlation_id,
        )
        bus.post_message(msg)

        def match(m):
            return m.get("correlation_id") == correlation_id and m.get("receiver_id") == "web_ui"

        reply = bus.get_message("web_ui", timeout=agent_config.chat_timeout_seconds, predicate=match)
        if reply is None:
            raise HTTPException(status_code=504, detail="Timeout waiting for agent response")
        if reply.get("status") == "error":
            detail = reply.get("payload", {}).get("error", "Unknown agent error")
            raise HTTPException(status_code=500, detail=detail)

        payload = reply.get("payload", {})
        text = (
            payload.get("data", {}).get("answer")
            or payload.get("summary")
            or payload.get("text")
            or "(No content)"
        )
        return ChatResponse(response=text, correlation_id=correlation_id)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"/chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
