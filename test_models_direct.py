#!/usr/bin/env python3
"""
Direct model tests:
- MedGemma agent (medical Q&A)
- Clinical Research agent (general clinical question and patient overview)
- Orchestrator model (raw LM Studio call for sanity)
"""

import asyncio
import logging
import httpx
import os
from uuid import uuid4
from a2a.client import ClientFactory, ClientConfig
from a2a.types import AgentCard, Message, Role, Part, TextPart, TransportProtocol


async def fetch_agent_card(base_url: str, httpx_client: httpx.AsyncClient) -> AgentCard:
    for path in ("/.well-known/agent.json", "/.well-known/agent-card.json"):
        try:
            resp = await httpx_client.get(f"{base_url}{path}")
            if resp.status_code == 200:
                return AgentCard(**resp.json())
        except Exception:
            continue
    raise RuntimeError(f"Could not fetch AgentCard from {base_url}")


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_medgemma_direct():
    logger.info("\n" + "=" * 60)
    logger.info("Direct: MedGemma")
    logger.info("=" * 60)

    httpx_client = httpx.AsyncClient(timeout=180)
    card = await fetch_agent_card("http://localhost:9101", httpx_client)
    client = ClientFactory(ClientConfig(
        httpx_client=httpx_client,
        supported_transports=[TransportProtocol.jsonrpc],
        use_client_preference=False,
    )).create(card)
    try:
        question = "What are common symptoms of hypertension?"
        msg = Message(role=Role.user, parts=[Part(root=TextPart(text=question))], messageId=str(uuid4()))
        last = None
        async for ev in client.send_message(msg):
            last = str(ev)
        logger.info(f"MedGemma last event: {(last or '<none>')[:400]}")
    finally:
        await client.close()
        await httpx_client.aclose()


async def test_clinical_direct():
    logger.info("\n" + "=" * 60)
    logger.info("Direct: Clinical Research")
    logger.info("=" * 60)

    httpx_client = httpx.AsyncClient(timeout=180)
    card = await fetch_agent_card("http://localhost:9102", httpx_client)
    try:
        client = ClientFactory(ClientConfig(
            httpx_client=httpx_client,
            supported_transports=[TransportProtocol.jsonrpc],
            use_client_preference=False,
        )).create(card)
    except Exception as e:
        logger.error(f"Skip Clinical direct test (client init failed): {e}")
        await httpx_client.aclose()
        return

    try:
        # General quick question
        q1 = "Explain what a randomized controlled trial is."
        msg1 = Message(role=Role.user, parts=[Part(root=TextPart(text=q1))], messageId=str(uuid4()))
        last1 = None
        async for ev in client.send_message(msg1):
            last1 = str(ev)
        logger.info(f"Clinical (general) last event: {(last1 or '<none>')[:400]}")

        # Patient overview prompt
        q2 = "Provide a high-level overview of available clinical data for patient 31f2e621-37c9-4e27-a87d-6689d678b7fd."
        msg2 = Message(role=Role.user, parts=[Part(root=TextPart(text=q2))], messageId=str(uuid4()))
        last2 = None
        async for ev in client.send_message(msg2):
            last2 = str(ev)
        logger.info(f"Clinical (patient overview) last event: {(last2 or '<none>')[:400]}")
    finally:
        await client.close()
        await httpx_client.aclose()


async def test_orchestrator_direct():
    logger.info("\n" + "=" * 60)
    logger.info("Direct: Orchestrator LM (raw LM Studio)")
    logger.info("=" * 60)

    base = os.getenv("LLM_BASE_URL", "http://localhost:1234")
    model = os.getenv("ORCHESTRATOR_MODEL", "meta-llama-3.1-8b-instruct")
    prompt = "Briefly explain what a router/orchestrator agent does in a multi-agent system."
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are a concise assistant."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
        "max_tokens": 400,
        "stream": False,
    }
    async with httpx.AsyncClient(timeout=60) as c:
        r = await c.post(f"{base}/v1/chat/completions", json=payload)
        r.raise_for_status()
        data = r.json()
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
        logger.info(f"Orchestrator LM reply: {text[:400]}")


async def main():
    logger.info("Waiting a moment for agents...")
    await asyncio.sleep(3)
    await test_medgemma_direct()
    await test_clinical_direct()
    await test_orchestrator_direct()


if __name__ == "__main__":
    import sys
    # Optional env-file loader for parity with other scripts
    env_file = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--env-file" and i + 1 < len(sys.argv[1:]):
            env_file = sys.argv[i + 2]
            break
    if env_file:
        from dotenv import load_dotenv
        load_dotenv(env_file)
    asyncio.run(main())


