#!/usr/bin/env python3
"""
MedGemma Agent Server
Runs the MedGemma medical Q&A agent as an A2A-compliant service
"""

import uvicorn
import logging
import json
from pathlib import Path
from dotenv import load_dotenv

from a2a.server.apps import A2AStarletteApplication
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCard, TransportProtocol
from .medgemma_executor import MedGemmaExecutor
from ..config import a2a_endpoints

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_medgemma_server():
    """Creates and returns the FastAPI application for the MedGemma agent."""
    
    # Load agent card from JSON for transparency
    card_path = Path(__file__).resolve().parents[1] / 'agent_cards' / 'medgemma.json'
    with card_path.open('r', encoding='utf-8') as f:
        card_data = json.load(f)
    # Ensure URL reflects runtime host/port from config
    agent_card = AgentCard(**card_data)
    agent_card.url = a2a_endpoints.medgemma_url
    # Ensure compatible transport preference
    agent_card.preferred_transport = TransportProtocol.jsonrpc
    
    # Create executor and request handler
    agent_executor = MedGemmaExecutor()
    request_handler = DefaultRequestHandler(
        agent_executor=agent_executor,
        task_store=InMemoryTaskStore()
    )
    
    # Create and run server (Starlette application exposes .well-known)
    server = A2AStarletteApplication(
        agent_card=agent_card,
        http_handler=request_handler
    )
    
    logger.info("MedGemma agent server application created.")
    return server.build()

app = create_medgemma_server()

if __name__ == '__main__':
    uvicorn.run(app, host="0.0.0.0", port=9101)
