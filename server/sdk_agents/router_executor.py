"""
Router Agent Executor for A2A SDK
Handles orchestration and routing to specialist agents.
"""

import httpx
import logging
import os
from typing import Dict, Any, Optional
import json

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    AgentCard,
    TextPart,
    Part,
    Task,
    TaskState,
    Message,
    Role,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.client import ClientConfig, ClientFactory
from a2a.client.card_resolver import A2ACardResolver
from a2a.types import TransportProtocol

logger = logging.getLogger(__name__)


class RouterAgentExecutor(AgentExecutor):
    """Router Agent Executor - orchestrates other agents."""
    
    def __init__(self):
        self.llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:1234")
        self.llm_api_key = os.getenv("LLM_API_KEY", "")
        self.orchestrator_model = os.getenv("ORCHESTRATOR_MODEL", "llama-3-8b-instruct")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
        self.http_client = httpx.AsyncClient(timeout=90.0)
        
        # Agent registry - in production this would be dynamic
        self.agents = {
            "medgemma": {
                "url": "http://localhost:9101",
                "name": "MedGemma Medical Assistant",
                "skills": ["answer_medical_question"]
            },
            "clinical": {
                "url": "http://localhost:9102", 
                "name": "Clinical Research Agent",
                "skills": ["clinical_research"]
            }
        }
        
        logger.info(f"Router agent executor initialized with model: {self.orchestrator_model}")
    
    async def _call_llm(self, messages: list[Dict[str, Any]]) -> str:
        """Call the LLM for routing decisions."""
        try:
            response = await self.http_client.post(
                f"{self.llm_base_url}/v1/chat/completions",
                json={
                    "model": self.orchestrator_model,
                    "messages": messages,
                    "temperature": self.temperature,
                    "max_tokens": 500
                },
                headers={"Authorization": f"Bearer {self.llm_api_key}"} if self.llm_api_key else {}
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            return "Error analyzing query"
    
    async def _route_query(self, query: str) -> Dict[str, Any]:
        """Determine which agent should handle the query."""
        
        # Create routing prompt
        agents_info = "\n".join([
            f"- {name}: {info['name']} (skills: {', '.join(info['skills'])})"
            for name, info in self.agents.items()
        ])
        
        system_prompt = f"""You are a query router for a medical multi-agent system.
Available agents:
{agents_info}

Analyze the query and determine which agent is best suited to handle it.
Respond with JSON: {{"agent": "agent_name", "reasoning": "why this agent"}}"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        response = await self._call_llm(messages)
        
        try:
            # Parse LLM response
            result = json.loads(response)
            agent_name = result.get("agent", "medgemma")
            reasoning = result.get("reasoning", "")
            
            if agent_name not in self.agents:
                agent_name = "medgemma"  # Default fallback
                
            return {
                "agent": agent_name,
                "reasoning": reasoning,
                "url": self.agents[agent_name]["url"]
            }
        except:
            # Default to medical agent on parse error
            return {
                "agent": "medgemma",
                "reasoning": "Default routing to medical agent",
                "url": self.agents["medgemma"]["url"]
            }
    
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute routing logic."""
        
        # Get the user query
        query = context.get_user_input()
        logger.info(f"Router received query: {query}")
        
        # Create task updater
        task_updater = TaskUpdater(event_queue, context.current_task.id if context.current_task else None, context.message.context_id if context.message else None)
        
        # Ensure a task exists and set running status
        task = context.current_task
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
            task_updater = TaskUpdater(event_queue, task.id, task.context_id)

        await task_updater.update_status(
            TaskState.working,
            new_agent_text_message(
                "Analyzing query and routing to appropriate agent...",
                task.context_id,
                task.id,
            ),
        )
        
        try:
            # Route the query
            routing_result = await self._route_query(query)
            logger.info(f"Routing to {routing_result['agent']}: {routing_result['reasoning']}")
            
            # Resolve agent card and create client per SDK samples
            agent_url = routing_result["url"]
            resolver = A2ACardResolver(self.http_client, agent_url)
            agent_card = await resolver.get_agent_card()

            client_config = ClientConfig(
                httpx_client=self.http_client,
                supported_transports=[
                    TransportProtocol.jsonrpc,
                    TransportProtocol.http_json,
                ],
                use_client_preference=True,
            )
            client = ClientFactory(client_config).create(agent_card)

            # Create message for target agent and send
            message = Message(role=Role.user, parts=[Part(root=TextPart(text=query))])

            response_text = ""
            async for event in client.send_message(message):
                # Unwrap tuple events
                evt = event[0] if isinstance(event, tuple) else event
                if hasattr(evt, "artifacts") and getattr(evt, "artifacts"):
                    for artifact in evt.artifacts:
                        for part in artifact.parts:
                            if hasattr(part, "root") and hasattr(part.root, "text"):
                                response_text = part.root.text
                if hasattr(evt, "status") and getattr(evt, "status") and getattr(evt.status, "message", None):
                    for part in evt.status.message.parts:
                        if hasattr(part, "root") and hasattr(part.root, "text"):
                            response_text = part.root.text
            
            
            if not response_text:
                response_text = "No response received from agent"
            
            # Send final response
            final_response = f"{response_text}\n\n*Handled by {routing_result['agent']}*"
            
            await task_updater.update_task(
                state=TaskState.completed,
                message=final_response
            )
            
        except Exception as e:
            logger.error(f"Router execution failed: {e}")
            await task_updater.update_task(
                state=TaskState.failed,
                message=f"Routing failed: {str(e)}"
            )
    
    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        """Handle task cancellation."""
        task_updater = TaskUpdater(
            event_queue,
            context.current_task.id if context.current_task else None,
            context.message.context_id if context.message else None,
        )
        
        await task_updater.update_task(
            state=TaskState.cancelled,
            message="Query routing cancelled"
        )
        
        return context.task
    
    # Agent card is provided via JSON in server/agent_cards/router.json
