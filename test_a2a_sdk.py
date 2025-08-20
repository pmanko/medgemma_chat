#!/usr/bin/env python3
"""
Test script for A2A SDK agents
Demonstrates using the official A2A SDK to communicate with agents
"""

import asyncio
import logging
from a2a.client import ClientFactory
from a2a.client.transports import JsonRpcTransport
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_agent_discovery():
    """Test agent discovery via A2A protocol"""
    logger.info("=" * 60)
    logger.info("Testing Agent Discovery")
    logger.info("=" * 60)
    
    agents = {
        "Router": "http://localhost:9100",
        "MedGemma": "http://localhost:9101",
        "Clinical": "http://localhost:9102"
    }
    
    for name, url in agents.items():
        try:
            client = ClientFactory.create_client(JsonRpcTransport(url=url))
            card = await client.get_agent_card()
            
            logger.info(f"\n{name} Agent:")
            logger.info(f"  Name: {card.name}")
            logger.info(f"  Description: {card.description}")
            logger.info(f"  Version: {getattr(card, 'version', 'N/A')}")
            logger.info(f"  Skills:")
            
            for skill in card.skills:
                logger.info(f"    - {skill.name}: {skill.description}")
                if hasattr(skill, 'input_schema'):
                    props = skill.input_schema.get('properties', {})
                    logger.info(f"      Inputs: {', '.join(props.keys())}")
                    
            await client.close()
            
        except Exception as e:
            logger.error(f"Failed to discover {name} agent: {e}")

async def test_medgemma_agent():
    """Test MedGemma medical Q&A"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing MedGemma Agent")
    logger.info("=" * 60)
    
    client = ClientFactory.create_client(JsonRpcTransport(url="http://localhost:9101"))
    
    queries = [
        "What are the common symptoms of hypertension?",
        "How is diabetes type 2 typically managed?",
        "What are the side effects of metformin?"
    ]
    
    for query in queries:
        try:
            logger.info(f"\nQuery: {query}")
            
            result = await client.invoke_skill(
                "answer_medical_question",
                query=query,
                include_references=True
            )
            
            logger.info(f"Answer: {result['answer'][:200]}...")
            logger.info(f"Confidence: {result.get('confidence', 'N/A')}")
            
            if 'references' in result and result['references']:
                logger.info(f"References: {', '.join(result['references'])}")
                
        except Exception as e:
            logger.error(f"Failed to query MedGemma: {e}")
    
    await client.close()

async def test_clinical_agent():
    """Test Clinical Research agent"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Clinical Research Agent")
    logger.info("=" * 60)
    
    client = ClientFactory.create_client(JsonRpcTransport(url="http://localhost:9102"))
    
    queries = [
        {
            "query": "Show me recent blood pressure readings for patients",
            "scope": "hie"
        },
        {
            "query": "What are the lab results for patient with diabetes?",
            "scope": "facility",
            "facility_id": "F001"
        }
    ]
    
    for q in queries:
        try:
            logger.info(f"\nQuery: {q['query']}")
            logger.info(f"Scope: {q.get('scope', 'hie')}")
            
            result = await client.invoke_skill(
                "clinical_research",
                **q
            )
            
            logger.info(f"Response: {result['response'][:200]}...")
            logger.info(f"Data Source: {result.get('data_source', 'N/A')}")
            logger.info(f"Records Found: {result.get('records_found', 'N/A')}")
            
        except Exception as e:
            logger.error(f"Failed to query Clinical agent: {e}")
    
    await client.close()

async def test_router_agent():
    """Test Router orchestration"""
    logger.info("\n" + "=" * 60)
    logger.info("Testing Router Agent (Orchestration)")
    logger.info("=" * 60)
    
    client = ClientFactory.create_client(JsonRpcTransport(url="http://localhost:9100"))
    
    test_cases = [
        {
            "query": "What are the symptoms of COVID-19?",
            "expected_agent": "medgemma"
        },
        {
            "query": "Show me the latest lab results for patient ID 12345",
            "expected_agent": "clinical"
        },
        {
            "query": "What medications are used to treat hypertension?",
            "expected_agent": "medgemma"
        },
        {
            "query": "Generate a report of all diabetic patients in facility F001",
            "expected_agent": "clinical",
            "scope": "facility",
            "facility_id": "F001"
        }
    ]
    
    for test in test_cases:
        try:
            query = test["query"]
            expected = test.get("expected_agent", "unknown")
            
            logger.info(f"\nQuery: {query}")
            logger.info(f"Expected routing: {expected}")
            
            # Build args
            args = {"query": query}
            if "scope" in test:
                args["scope"] = test["scope"]
            if "facility_id" in test:
                args["facility_id"] = test["facility_id"]
            
            result = await client.invoke_skill("route_query", **args)
            
            agent_used = result.get("agent_used", "unknown")
            skill_used = result.get("skill_used", "unknown")
            confidence = result.get("routing_confidence", 0)
            
            logger.info(f"Routed to: {agent_used}.{skill_used}")
            logger.info(f"Confidence: {confidence}")
            logger.info(f"Response: {result['response'][:200]}...")
            
            if agent_used == expected:
                logger.info("✓ Routing matched expectation")
            else:
                logger.warning(f"✗ Expected {expected}, got {agent_used}")
                
        except Exception as e:
            logger.error(f"Failed to query Router: {e}")
    
    await client.close()

async def test_end_to_end():
    """Test complete flow through the system"""
    logger.info("\n" + "=" * 60)
    logger.info("End-to-End Test via A2A Protocol")
    logger.info("=" * 60)
    
    # Simulate a conversation
    conversation = [
        "What is hypertension?",
        "What are the latest blood pressure readings in our facility?",
        "What medications are typically prescribed for high blood pressure?",
        "Show me patients with uncontrolled hypertension"
    ]
    
    router_client = A2AClient("http://localhost:9100")
    conversation_id = "test-conversation-001"
    
    for i, query in enumerate(conversation, 1):
        try:
            logger.info(f"\n[Turn {i}] User: {query}")
            
            result = await router_client.invoke_skill(
                "route_query",
                query=query,
                conversation_id=conversation_id,
                scope="hie"
            )
            
            logger.info(f"[Turn {i}] Agent: {result['agent_used']}")
            logger.info(f"[Turn {i}] Response: {result['response'][:300]}...")
            
        except Exception as e:
            logger.error(f"Failed at turn {i}: {e}")
    
    await router_client.close()

async def main():
    """Run all tests"""
    
    # Wait for agents to be ready
    logger.info("Waiting for agents to start...")
    await asyncio.sleep(2)
    
    try:
        # Run tests in sequence
        await test_agent_discovery()
        await test_medgemma_agent()
        await test_clinical_agent()
        await test_router_agent()
        await test_end_to_end()
        
        logger.info("\n" + "=" * 60)
        logger.info("All tests completed!")
        logger.info("=" * 60)
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}", exc_info=True)

if __name__ == "__main__":
    import sys
    
    # Check for --env-file parameter
    env_file = ".env"
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--env-file" and i + 1 < len(sys.argv[1:]):
            env_file = sys.argv[i + 2]
            break
    
    # Load environment from specified file
    from dotenv import load_dotenv
    load_dotenv(env_file)
    
    asyncio.run(main())
