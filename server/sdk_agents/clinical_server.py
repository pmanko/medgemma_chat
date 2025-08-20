"""
Clinical Research Agent Server for A2A SDK
Sets up the FastAPI application for the Clinical Research agent.
"""

import uvicorn
import logging
import json
from pathlib import Path
from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, TransportProtocol, AgentCard
from .clinical_executor import ClinicalExecutor

logger = logging.getLogger(__name__)


def create_clinical_server():
    """
    Creates and returns the FastAPI application for the Clinical Research agent.
    """
    agent_executor = ClinicalExecutor()
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore()
    )
    
    # Load agent card from JSON for transparency
    card_path = Path(__file__).resolve().parents[1] / 'agent_cards' / 'clinical.json'
    with card_path.open('r', encoding='utf-8') as f:
        card_data = json.load(f)
    agent_card = AgentCard(**card_data)
    # Ensure runtime URL
    agent_card.url = "http://localhost:9102/"
    agent_card.preferred_transport = TransportProtocol.http_json
    agent_card.capabilities = AgentCapabilities(streaming=True)
    
    # Create the server application (Starlette to expose .well-known)
    server_app = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )
    
    logger.info("Clinical Research A2A server application created.")
    return server_app.build()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    app = create_clinical_server()
    uvicorn.run(app, host="0.0.0.0", port=9102)
