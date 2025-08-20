#!/usr/bin/env python3
"""
Launch script for A2A SDK-based agents
Starts all three agents as separate services
"""

import asyncio
import sys
import os
import subprocess
import signal
import logging
from typing import List
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global list to track subprocesses
processes: List[subprocess.Popen] = []

def signal_handler(sig, frame):
    """Handle shutdown signals"""
    logger.info(f"Received signal {sig}, shutting down agents...")
    for proc in processes:
        if proc.poll() is None:  # Process is still running
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
    sys.exit(0)

def start_agent(name: str, module: str, port: int) -> subprocess.Popen:
    """Start an agent as a subprocess"""
    logger.info(f"Starting {name} on port {port}")
    
    # Use poetry run to ensure correct environment
    cmd = [
        "poetry", "run", "python", "-m",
        module,
        "--host", "0.0.0.0",
        "--port", str(port)
    ]
    
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    processes.append(proc)
    return proc

async def check_agent_health(url: str, name: str, max_retries: int = 10):
    """Check if an agent is responding"""
    import httpx
    
    for i in range(max_retries):
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{url}/.well-known/agent.json")
                if response.status_code == 200:
                    data = response.json()
                    logger.info(f"✓ {name} agent ready: {data.get('name', 'Unknown')}")
                    return True
        except Exception:
            await asyncio.sleep(2)
    
    logger.error(f"✗ {name} agent failed to start")
    return False

async def main():
    """Main entry point"""
    
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Start agents
    agents = [
        ("MedGemma", "server.sdk_agents.medgemma_server", 9101),
        ("Clinical", "server.sdk_agents.clinical_server", 9102),
        ("Router", "server.sdk_agents.router_server", 9100),
    ]
    
    for name, module, port in agents:
        start_agent(name, module, port)
    
    # Wait a bit for agents to start
    await asyncio.sleep(3)
    
    # Check agent health
    health_checks = [
        check_agent_health(f"http://localhost:{port}", name)
        for name, _, port in agents
    ]
    
    results = await asyncio.gather(*health_checks)
    
    if all(results):
        logger.info("=" * 60)
        logger.info("All agents started successfully!")
        logger.info("=" * 60)
        logger.info("Agent endpoints:")
        for name, _, port in agents:
            logger.info(f"  {name}: http://localhost:{port}")
        logger.info("=" * 60)
        logger.info("Press Ctrl+C to stop all agents")
        
        # Keep running until interrupted
        try:
            while True:
                await asyncio.sleep(1)
                # Check if any process has died
                for proc, (name, _, _) in zip(processes, agents):
                    if proc.poll() is not None:
                        logger.error(f"{name} agent has stopped unexpectedly")
                        signal_handler(signal.SIGTERM, None)
        except KeyboardInterrupt:
            pass
    else:
        logger.error("Some agents failed to start")
        signal_handler(signal.SIGTERM, None)

if __name__ == "__main__":
    # Check for --env-file parameter
    env_file = ".env"
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--env-file" and i + 1 < len(sys.argv[1:]):
            env_file = sys.argv[i + 2]
            # Remove these args so they don't get passed to subprocesses
            sys.argv.pop(i + 1)
            sys.argv.pop(i + 1)
            break
    
    # Load environment from specified file
    load_dotenv(env_file)
    
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
    orchestrator_provider = os.getenv('ORCHESTRATOR_PROVIDER') or 'local'
    orchestrator_model = os.getenv('ORCHESTRATOR_MODEL', 'Not set')
    logger.info(f"Orchestrator Provider: {orchestrator_provider}")
    logger.info(f"Orchestrator Model: {orchestrator_model}")
    logger.info("=" * 60)
    
    # Run the main async function
    try:
        asyncio.run(main())
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)