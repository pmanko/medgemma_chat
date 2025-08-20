"""
Router Agent Server for A2A SDK
Sets up the FastAPI application for the Router agent.
"""

import uvicorn
import logging
import json
from pathlib import Path
from a2a.server.apps import A2ARESTFastAPIApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, TransportProtocol, AgentCard
from .router_executor import RouterAgentExecutor

logger = logging.getLogger(__name__)


def create_router_server():
    """
    Creates and returns the FastAPI application for the Router agent.
    """
    agent_executor = RouterAgentExecutor()
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore()
    )
    
    # Load agent card from JSON for transparency
    card_path = Path(__file__).resolve().parents[1] / 'agent_cards' / 'router.json'
    with card_path.open('r', encoding='utf-8') as f:
        card_data = json.load(f)
    agent_card = AgentCard(**card_data)
    # Ensure runtime URL
    agent_card.url = "http://localhost:9100/"
    agent_card.preferred_transport = TransportProtocol.http_json
    agent_card.capabilities = AgentCapabilities(streaming=True)
    
    # Create the server application
    server_app = A2ARESTFastAPIApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )
    
    logger.info("Router A2A server application created.")
    return server_app.build()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_router_server()
    uvicorn.run(app, host="0.0.0.0", port=9100)
