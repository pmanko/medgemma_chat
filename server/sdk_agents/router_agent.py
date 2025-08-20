"""
Router Agent using A2A SDK
Orchestrates requests to appropriate specialist agents
"""

from a2a import Agent
from a2a.types import AgentCard, Skill, InputSchema, OutputSchema
from a2a.decorators import skill
from a2a.client import AgentClient
import httpx
import logging
import json
from typing import Dict, Any, Optional, List
import os

logger = logging.getLogger(__name__)

class RouterAgent(Agent):
    """Orchestrator that routes requests to appropriate specialist agents"""
    
    def __init__(self):
        super().__init__()
        
        # Orchestrator LLM configuration
        self.orchestrator_provider = os.getenv("ORCHESTRATOR_PROVIDER", "openai")
        self.orchestrator_model = os.getenv("ORCHESTRATOR_MODEL", "llama-3-8b-instruct")
        self.gemini_api_key = os.getenv("GEMINI_API_KEY", "")
        self.llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:1234")
        self.llm_api_key = os.getenv("LLM_API_KEY", "")
        
        # Agent URLs
        self.agent_urls = {
            "medgemma": os.getenv("A2A_MEDGEMMA_URL", "http://localhost:9101"),
            "clinical": os.getenv("A2A_CLINICAL_URL", "http://localhost:9102")
        }
        
        # Initialize agent clients
        self.agent_clients = {}
        for name, url in self.agent_urls.items():
            try:
                self.agent_clients[name] = AgentClient(url)
                logger.info(f"Initialized client for {name} agent at {url}")
            except Exception as e:
                logger.error(f"Failed to initialize {name} agent client: {e}")
        
        self.http_client = httpx.AsyncClient(timeout=90.0)
        logger.info(f"Router agent initialized with orchestrator: {self.orchestrator_provider}/{self.orchestrator_model}")
    
    async def get_agent_card(self) -> AgentCard:
        """Return agent capabilities following A2A protocol"""
        return AgentCard(
            name="Medical Query Router",
            description="Routes medical queries to appropriate specialist agents and returns unified responses",
            version="1.0.0",
            skills=[
                Skill(
                    name="route_query",
                    description="Analyze query and route to the best specialist agent",
                    input_schema=InputSchema(
                        type="object",
                        properties={
                            "query": {
                                "type": "string",
                                "description": "User query to route"
                            },
                            "conversation_id": {
                                "type": "string",
                                "description": "Conversation identifier for context"
                            },
                            "scope": {
                                "type": "string",
                                "enum": ["facility", "hie"],
                                "default": "hie",
                                "description": "Access scope for clinical data"
                            },
                            "facility_id": {
                                "type": "string",
                                "description": "Facility identifier"
                            },
                            "org_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Organization identifiers"
                            }
                        },
                        required=["query"]
                    ),
                    output_schema=OutputSchema(
                        type="object",
                        properties={
                            "response": {"type": "string", "description": "Final response from specialist"},
                            "agent_used": {"type": "string", "description": "Which agent handled the query"},
                            "skill_used": {"type": "string", "description": "Which skill was invoked"},
                            "routing_confidence": {"type": "number", "description": "Confidence in routing decision"}
                        },
                        required=["response", "agent_used", "skill_used"]
                    )
                )
            ]
        )
    
    @skill("route_query")
    async def route_query(
        self,
        query: str,
        conversation_id: Optional[str] = None,
        scope: str = "hie",
        facility_id: Optional[str] = None,
        org_ids: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Analyze query and route to the best specialist agent
        
        Args:
            query: User query to route
            conversation_id: Conversation context
            scope: Access scope for clinical data
            facility_id: Facility identifier
            org_ids: Organization identifiers
            
        Returns:
            Response from the selected specialist agent
        """
        logger.info(f"Routing query: {query[:100]}... (conversation: {conversation_id})")
        
        # Discover available agents and their capabilities
        available_agents = await self._discover_agents()
        
        if not available_agents:
            logger.error("No agents available for routing")
            return {
                "response": "No specialist agents are currently available. Please try again later.",
                "agent_used": "none",
                "skill_used": "none",
                "routing_confidence": 0.0
            }
        
        # Use orchestrator to select best agent and skill
        routing_decision = await self._select_agent(
            query, 
            available_agents,
            {"scope": scope, "facility_id": facility_id, "org_ids": org_ids}
        )
        
        selected_agent = routing_decision["agent"]
        selected_skill = routing_decision["skill"]
        skill_args = routing_decision["args"]
        confidence = routing_decision.get("confidence", 0.8)
        
        logger.info(f"Routing to {selected_agent}.{selected_skill} with confidence {confidence}")
        
        # Invoke the selected agent's skill
        try:
            client = self.agent_clients[selected_agent]
            result = await client.invoke_skill(selected_skill, **skill_args)
            
            # Extract response based on agent type
            if selected_agent == "medgemma":
                response = result.get("answer", result.get("response", str(result)))
            elif selected_agent == "clinical":
                response = result.get("response", str(result))
            else:
                response = str(result)
            
            return {
                "response": response,
                "agent_used": selected_agent,
                "skill_used": selected_skill,
                "routing_confidence": confidence
            }
            
        except Exception as e:
            logger.error(f"Failed to invoke {selected_agent}.{selected_skill}: {e}")
            
            # Fallback to other agent if available
            fallback_agent = "medgemma" if selected_agent == "clinical" else "clinical"
            if fallback_agent in self.agent_clients:
                logger.info(f"Attempting fallback to {fallback_agent}")
                try:
                    client = self.agent_clients[fallback_agent]
                    # Use generic skill for fallback
                    if fallback_agent == "medgemma":
                        result = await client.invoke_skill("answer_medical_question", query=query)
                        response = result.get("answer", "Unable to process query")
                    else:
                        result = await client.invoke_skill("clinical_research", query=query, scope=scope)
                        response = result.get("response", "Unable to process query")
                    
                    return {
                        "response": response,
                        "agent_used": fallback_agent,
                        "skill_used": "fallback",
                        "routing_confidence": 0.5
                    }
                except Exception as fallback_error:
                    logger.error(f"Fallback also failed: {fallback_error}")
            
            return {
                "response": f"Error processing query: {str(e)}",
                "agent_used": selected_agent,
                "skill_used": selected_skill,
                "routing_confidence": 0.0
            }
    
    async def _discover_agents(self) -> Dict[str, Any]:
        """Discover available agents and their capabilities"""
        available = {}
        
        for name, client in self.agent_clients.items():
            try:
                card = await client.get_agent_card()
                available[name] = {
                    "card": card,
                    "skills": [
                        {
                            "name": skill.name,
                            "description": skill.description,
                            "input_schema": skill.input_schema
                        }
                        for skill in card.skills
                    ]
                }
                logger.info(f"Discovered {name} agent with {len(card.skills)} skills")
            except Exception as e:
                logger.warning(f"Could not discover {name} agent: {e}")
        
        return available
    
    async def _select_agent(
        self, 
        query: str, 
        available_agents: Dict[str, Any],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Use orchestrator LLM to select best agent and skill"""
        
        # Build capabilities description
        capabilities = []
        for agent_name, agent_info in available_agents.items():
            for skill in agent_info["skills"]:
                capabilities.append({
                    "agent": agent_name,
                    "skill": skill["name"],
                    "description": skill["description"],
                    "input_properties": list(skill["input_schema"].get("properties", {}).keys())
                })
        
        # Create routing prompt
        prompt = f"""You are a medical query router. Select the best agent and skill for this query.

User Query: {query}

Available Capabilities:
{json.dumps(capabilities, indent=2)}

Context:
- Scope: {context.get('scope', 'hie')}
- Facility ID: {context.get('facility_id', 'none')}
- Organizations: {context.get('org_ids', [])}

Rules:
1. Use 'medgemma' agent for general medical questions, symptoms, treatments, medications
2. Use 'clinical' agent for patient data, lab results, FHIR queries, statistics, reports
3. Include appropriate arguments based on the skill's input_properties

Return ONLY valid JSON in this format:
{{
    "agent": "agent_name",
    "skill": "skill_name",
    "args": {{"query": "...", "other_args": "..."}},
    "confidence": 0.0-1.0
}}"""
        
        # Call orchestrator
        if self.orchestrator_provider == "gemini":
            response_text = await self._call_gemini(prompt)
        else:
            response_text = await self._call_openai_compatible(prompt)
        
        # Parse response
        try:
            # Clean up response if needed
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            decision = json.loads(response_text.strip())
            
            # Validate and set defaults
            if "agent" not in decision or decision["agent"] not in available_agents:
                # Fallback logic
                if any(kw in query.lower() for kw in ["patient", "lab", "result", "data", "fhir"]):
                    decision["agent"] = "clinical"
                    decision["skill"] = "clinical_research"
                else:
                    decision["agent"] = "medgemma"
                    decision["skill"] = "answer_medical_question"
            
            # Ensure args includes query
            if "args" not in decision:
                decision["args"] = {}
            if "query" not in decision["args"]:
                decision["args"]["query"] = query
            
            # Add context to args if clinical agent
            if decision["agent"] == "clinical":
                decision["args"].update({
                    "scope": context.get("scope", "hie"),
                    "facility_id": context.get("facility_id"),
                    "org_ids": context.get("org_ids")
                })
            
            # Set default confidence if not provided
            if "confidence" not in decision:
                decision["confidence"] = 0.8
            
            return decision
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse routing decision: {e}, Response: {response_text[:200]}")
            
            # Fallback routing based on keywords
            if any(kw in query.lower() for kw in ["patient", "lab", "result", "data", "fhir", "observation"]):
                return {
                    "agent": "clinical",
                    "skill": "clinical_research",
                    "args": {
                        "query": query,
                        "scope": context.get("scope", "hie"),
                        "facility_id": context.get("facility_id"),
                        "org_ids": context.get("org_ids")
                    },
                    "confidence": 0.6
                }
            else:
                return {
                    "agent": "medgemma",
                    "skill": "answer_medical_question",
                    "args": {"query": query},
                    "confidence": 0.6
                }
    
    async def _call_openai_compatible(self, prompt: str) -> str:
        """Call OpenAI-compatible LLM endpoint"""
        headers = {"Content-Type": "application/json"}
        if self.llm_api_key:
            headers["Authorization"] = f"Bearer {self.llm_api_key}"
        
        response = await self.http_client.post(
            f"{self.llm_base_url}/v1/chat/completions",
            json={
                "model": self.orchestrator_model,
                "messages": [
                    {"role": "system", "content": "You are a routing assistant. Return only valid JSON."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 200
            },
            headers=headers
        )
        
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    
    async def _call_gemini(self, prompt: str) -> str:
        """Call Google Gemini API"""
        if not self.gemini_api_key:
            logger.warning("Gemini API key not configured, falling back to OpenAI-compatible")
            return await self._call_openai_compatible(prompt)
        
        response = await self.http_client.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.orchestrator_model}:generateContent",
            json={
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {
                    "temperature": 0.1,
                    "maxOutputTokens": 200
                }
            },
            headers={"x-goog-api-key": self.gemini_api_key}
        )
        
        response.raise_for_status()
        data = response.json()
        return data["candidates"][0]["content"]["parts"][0]["text"].strip()
    
    async def cleanup(self):
        """Cleanup resources on shutdown"""
        await self.http_client.aclose()
        for client in self.agent_clients.values():
            try:
                await client.close()
            except:
                pass
        logger.info("Router agent cleanup completed")


# For standalone testing
if __name__ == "__main__":
    import asyncio
    from a2a.server import AgentServer
    import uvicorn
    
    async def main():
        agent = RouterAgent()
        server = AgentServer(agent)
        
        # Test the agent
        card = await agent.get_agent_card()
        print(f"Agent: {card.name}")
        print(f"Skills: {[s.name for s in card.skills]}")
        
        # Run server
        config = uvicorn.Config(
            server.app,
            host="0.0.0.0",
            port=9100,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    asyncio.run(main())
