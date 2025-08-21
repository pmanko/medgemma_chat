"""
Clinical Research Agent Executor using A2A SDK
Handles clinical data queries and research questions
"""

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import (
    AgentCard,
    AgentCapabilities,
    AgentSkill,
    Part,
    TextPart,
    Task,
    TaskState,
    UnsupportedOperationError,
)
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError
import httpx
import logging
import os
import json

logger = logging.getLogger(__name__)


class ClinicalExecutor(AgentExecutor):
    """Clinical research agent executor"""
    
    def __init__(self):
        self.llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:1234")
        self.llm_api_key = os.getenv("LLM_API_KEY", "")
        self.general_model = os.getenv("CLINICAL_RESEARCH_MODEL", 'gemma-3-1b-it')
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.3"))
        self.http_client = httpx.AsyncClient(timeout=180.0)
        
        # Clinical data sources (optional)
        self.fhir_base_url = os.getenv("OPENMRS_FHIR_BASE_URL")
        self.fhir_username = os.getenv("OPENMRS_USERNAME")
        self.fhir_password = os.getenv("OPENMRS_PASSWORD")
        
        logger.info(f"Clinical executor initialized with model: {self.general_model}")
    
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute clinical research request"""
        query = context.get_user_input()
        task = context.current_task
        
        # Create a new task if none exists
        if not task:
            task = new_task(context.message)
            await event_queue.enqueue_event(task)
        
        updater = TaskUpdater(event_queue, task.id, task.context_id)
        
        try:
            # Update status to working
            await updater.update_status(
                TaskState.working,
                new_agent_text_message("Analyzing clinical research question...", task.context_id, task.id)
            )
            
            # Prepare system prompt
            system_prompt = """You are a Clinical Research Assistant specializing in evidence-based medicine and clinical data analysis.
            
            Your expertise includes:
            1. Clinical trial design and methodology
            2. Epidemiological research
            3. Patient data analysis and statistics
            4. Medical literature review
            5. Clinical guidelines and protocols
            
            Important guidelines:
            1. Cite relevant studies when possible
            2. Explain statistical concepts clearly
            3. Discuss limitations of research findings
            4. Maintain patient privacy and HIPAA compliance
            5. Provide evidence-based recommendations"""
            
            # Add context about available data sources if configured
            if self.fhir_base_url:
                system_prompt += "\n\nNote: You have access to clinical data from a FHIR server, though direct queries are not implemented in this demo."
            
            # Call LLM
            headers = {"Content-Type": "application/json"}
            if self.llm_api_key:
                headers["Authorization"] = f"Bearer {self.llm_api_key}"
            
            request_data = {
                "model": self.general_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                "temperature": self.temperature,
                "max_tokens": 1500
            }
            
            response = await self.http_client.post(
                f"{self.llm_base_url}/v1/chat/completions",
                headers=headers,
                json=request_data
            )
            response.raise_for_status()
            
            result = response.json()
            answer = result.get("choices", [{}])[0].get("message", {}).get("content", "")
            
            if not answer:
                answer = "I apologize, but I couldn't generate a response. Please try rephrasing your question."
            
            # Add research disclaimer if not present
            if "research" not in answer.lower() and "study" not in answer.lower():
                answer += "\n\n**Note:** This analysis is based on general clinical research principles. For specific patient care decisions, always consult current clinical guidelines and patient-specific factors."
            
            # Add the response as an artifact
            await updater.add_artifact(
                [Part(root=TextPart(text=answer))],
                name='clinical_analysis'
            )
            
            # Complete the task
            await updater.complete()
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling LLM: {e}")
            error_msg = f"Error processing clinical research question: {str(e)}"
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(error_msg, task.context_id, task.id)
            )
            
        except Exception as e:
            logger.error(f"Error processing clinical query: {e}", exc_info=True)
            error_msg = f"Error processing clinical research question: {str(e)}"
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(error_msg, task.context_id, task.id)
            )
    
    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        """Cancel is not supported for this agent"""
        raise ServerError(error=UnsupportedOperationError(
            message="Cancel operation is not supported for Clinical agent"
        ))
    
    async def cleanup(self):
        """Clean up resources"""
        await self.http_client.aclose()
        logger.info("Clinical executor cleanup completed")

    def get_agent_card(self) -> AgentCard:
        """Return agent capabilities for A2A discovery."""
        return AgentCard(
            name="Clinical Research Agent",
            description="Provides clinical research analysis and general clinical answers",
            url="http://localhost:9102/",
            version="1.0.0",
            default_input_modes=["text", "text/plain"],
            default_output_modes=["text", "text/plain"],
            capabilities=AgentCapabilities(streaming=True),
            skills=[
                AgentSkill(
                    id="clinical_research",
                    name="clinical_research",
                    description="Retrieve and analyze clinical data with scope-based access",
                    tags=["clinical", "research"],
                    input_schema={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Clinical question"},
                            "scope": {
                                "type": "string",
                                "enum": ["facility", "hie"],
                                "default": "hie",
                            },
                            "facility_id": {"type": "string"},
                            "org_ids": {"type": "array", "items": {"type": "string"}},
                            "data_source": {
                                "type": "string",
                                "enum": ["auto", "fhir", "sql"],
                                "default": "auto",
                            },
                        },
                        "required": ["query"],
                    },
                    output_schema={
                        "type": "object",
                        "properties": {
                            "response": {"type": "string"},
                            "data_source": {"type": "string"},
                            "scope": {"type": "string"},
                            "records_found": {"type": "integer"},
                            "query_executed": {"type": "string"},
                        },
                    },
                )
            ],
        )
