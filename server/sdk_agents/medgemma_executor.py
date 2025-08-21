"""
MedGemma Agent Executor using A2A SDK
Handles medical Q&A requests using the MedGemma model
"""

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import Part, TextPart, Task, TaskState, UnsupportedOperationError
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError
import httpx
import logging
import os
import json

logger = logging.getLogger(__name__)


class MedGemmaExecutor(AgentExecutor):
    """Medical Q&A agent executor using MedGemma model"""
    
    def __init__(self):
        self.llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:1234")
        self.llm_api_key = os.getenv("LLM_API_KEY", "")
        self.med_model = os.getenv("MED_MODEL", "medgemma-2")
        self.temperature = float(os.getenv("LLM_TEMPERATURE", "0.1"))
        self.http_client = httpx.AsyncClient(timeout=180.0)
        logger.info(f"MedGemma executor initialized with model: {self.med_model}")
    
    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute medical Q&A request"""
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
                new_agent_text_message("Processing your medical question...", task.context_id, task.id)
            )
            
            # Prepare system prompt
            system_prompt = """You are MedGemma, a medical AI assistant trained on clinical literature. 
            Provide accurate, evidence-based medical information. 
            Be clear about limitations and always recommend consulting healthcare professionals for personal medical advice.
            
            Important guidelines:
            1. Provide evidence-based information
            2. Use clear, patient-friendly language
            3. Include appropriate medical disclaimers
            4. Never diagnose or prescribe treatments
            5. Recommend professional consultation when appropriate"""
            
            # Call LLM
            headers = {"Content-Type": "application/json"}
            if self.llm_api_key:
                headers["Authorization"] = f"Bearer {self.llm_api_key}"
            
            request_data = {
                "model": self.med_model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": query}
                ],
                "temperature": self.temperature,
                "max_tokens": 1000
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
            
            # Add disclaimer if not already present
            if "consult" not in answer.lower() and "professional" not in answer.lower():
                answer += "\n\n**Disclaimer:** This information is for educational purposes only and should not replace professional medical advice. Always consult with a qualified healthcare provider for medical concerns."
            
            # Add the response as an artifact
            await updater.add_artifact(
                [Part(root=TextPart(text=answer))],
                name='medical_response'
            )
            
            # Complete the task
            await updater.complete()
            
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error calling LLM: {e}")
            error_msg = f"I encountered an error processing your medical question. Please try again. Error: {str(e)}"
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(error_msg, task.context_id, task.id)
            )
            
        except Exception as e:
            logger.error(f"Error processing medical query: {e}", exc_info=True)
            error_msg = f"I encountered an error processing your medical question. Please try again. Error: {str(e)}"
            await updater.update_status(
                TaskState.failed,
                new_agent_text_message(error_msg, task.context_id, task.id)
            )
    
    async def cancel(
        self, context: RequestContext, event_queue: EventQueue
    ) -> Task | None:
        """Cancel is not supported for this agent"""
        raise ServerError(error=UnsupportedOperationError(
            message="Cancel operation is not supported for MedGemma agent"
        ))
    
    async def cleanup(self):
        """Clean up resources"""
        await self.http_client.aclose()
        logger.info("MedGemma executor cleanup completed")
