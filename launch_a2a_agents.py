#!/usr/bin/env python3
"""
Launch script for A2A SDK-based agents
Starts all three agents as separate services
"""

import asyncio
import sys
import os
import signal
import logging
from typing import Dict, Any
import uvicorn
from a2a.server import AgentServer

# Add server directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from server.sdk_agents import MedGemmaAgent, ClinicalResearchAgent, RouterAgent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
shutdown_event = asyncio.Event()

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {sig}, initiating graceful shutdown...")
    shutdown_event.set()

async def start_medgemma_agent(port: int = 9101):
    """Start MedGemma agent server"""
    logger.info(f"Starting MedGemma agent on port {port}")
    agent = MedGemmaAgent()
    server = AgentServer(agent)
    
    config = uvicorn.Config(
        server.app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=False
    )
    server_instance = uvicorn.Server(config)
    
    # Run until shutdown
    await server_instance.serve()
    await agent.cleanup()

async def start_clinical_agent(port: int = 9102):
    """Start Clinical Research agent server"""
    logger.info(f"Starting Clinical Research agent on port {port}")
    agent = ClinicalResearchAgent()
    server = AgentServer(agent)
    
    config = uvicorn.Config(
        server.app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=False
    )
    server_instance = uvicorn.Server(config)
    
    # Run until shutdown
    await server_instance.serve()
    await agent.cleanup()

async def start_router_agent(port: int = 9100):
    """Start Router agent server"""
    logger.info(f"Starting Router agent on port {port}")
    
    # Wait a bit for other agents to start
    await asyncio.sleep(2)
    
    agent = RouterAgent()
    server = AgentServer(agent)
    
    config = uvicorn.Config(
        server.app,
        host="0.0.0.0",
        port=port,
        log_level="info",
        access_log=False
    )
    server_instance = uvicorn.Server(config)
    
    # Run until shutdown
    await server_instance.serve()
    await agent.cleanup()

async def test_agents():
    """Test that all agents are responding"""
    import httpx
    
    await asyncio.sleep(3)  # Wait for agents to start
    
    async with httpx.AsyncClient() as client:
        agents = [
            ("Router", "http://localhost:9100"),
            ("MedGemma", "http://localhost:9101"),
            ("Clinical", "http://localhost:9102")
        ]
        
        for name, url in agents:
            try:
                # Test agent card endpoint
                response = await client.post(
                    url,
                    json={
                        "jsonrpc": "2.0",
                        "method": "get_agent_card",
                        "params": {},
                        "id": 1
                    }
                )
                if response.status_code == 200:
                    data = response.json()
                    if "result" in data:
                        card = data["result"]
                        logger.info(f"✓ {name} agent ready: {card.get('name', 'Unknown')}")
                    else:
                        logger.warning(f"⚠ {name} agent responded but no card: {data}")
                else:
                    logger.error(f"✗ {name} agent error: HTTP {response.status_code}")
            except Exception as e:
                logger.error(f"✗ {name} agent not responding: {e}")

async def main():
    """Main entry point"""
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Parse command line arguments
    if len(sys.argv) > 1:
        agent_type = sys.argv[1].lower()
        
        if agent_type == "medgemma":
            await start_medgemma_agent()
        elif agent_type == "clinical":
            await start_clinical_agent()
        elif agent_type == "router":
            await start_router_agent()
        elif agent_type == "test":
            await test_agents()
        else:
            logger.error(f"Unknown agent type: {agent_type}")
            print("Usage: python launch_a2a_agents.py [medgemma|clinical|router|all|test]")
            sys.exit(1)
    else:
        # Start all agents concurrently
        logger.info("Starting all A2A agents...")
        
        # Create tasks for all agents
        tasks = [
            asyncio.create_task(start_medgemma_agent()),
            asyncio.create_task(start_clinical_agent()),
            asyncio.create_task(start_router_agent()),
        ]
        
        # Also run a test after startup
        asyncio.create_task(test_agents())
        
        try:
            # Wait for shutdown signal
            await shutdown_event.wait()
            
            # Cancel all tasks
            for task in tasks:
                task.cancel()
            
            # Wait for tasks to complete
            await asyncio.gather(*tasks, return_exceptions=True)
            
        except KeyboardInterrupt:
            logger.info("Keyboard interrupt received")
        finally:
            logger.info("All agents stopped")

if __name__ == "__main__":
    # Check for required environment variables
    required_vars = ["LLM_BASE_URL"]
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.info("Please set these in your .env file or environment")
        sys.exit(1)
    
    # Display configuration
    logger.info("=" * 60)
    logger.info("A2A Multi-Agent Medical Chat System")
    logger.info("=" * 60)
    logger.info(f"LLM Base URL: {os.getenv('LLM_BASE_URL')}")
    logger.info(f"General Model: {os.getenv('GENERAL_MODEL', 'Not set')}")
    logger.info(f"Medical Model: {os.getenv('MED_MODEL', 'Not set')}")
    logger.info(f"Orchestrator: {os.getenv('ORCHESTRATOR_PROVIDER', 'openai')}")
    logger.info(f"FHIR Server: {os.getenv('OPENMRS_FHIR_BASE_URL', 'Not configured')}")
    logger.info(f"Spark SQL: {os.getenv('SPARK_THRIFT_HOST', 'Not configured')}")
    logger.info("=" * 60)
    
    # Run the main async function
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)
