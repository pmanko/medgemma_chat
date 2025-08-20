# Migration Guide: From Simulated A2A to Official SDK

## Overview

This guide walks through migrating the multiagent_chat project from its custom simulated A2A implementation to using the official [A2A Python SDK](https://github.com/a2aproject/a2a-samples).

## Prerequisites

1. **Install the A2A SDK**:
```bash
pip install a2a-sdk
# or add to pyproject.toml:
# a2a-sdk = "^0.3.0"
```

2. **Remove custom A2A layer**:
- Delete `server/a2a_layer.py` (AgentRegistry, MessageBus)
- Remove threading-based agent loops
- Remove custom message format handling

## Key Differences

| Aspect | Current (Simulated) | A2A SDK |
|--------|-------------------|----------|
| **Agent Definition** | Custom classes with threads | `a2a.Agent` base class |
| **Message Format** | Custom JSON | JSON-RPC 2.0 |
| **Discovery** | In-memory registry | Agent Cards via HTTP |
| **Communication** | Python Queue | HTTP/HTTPS endpoints |
| **Deployment** | Single process or services | Always separate services |

## Migration Steps

### Step 1: Update Dependencies

```toml
# pyproject.toml
[tool.poetry.dependencies]
python = ">=3.10,<3.13"  # SDK requires 3.10+
a2a-sdk = "^0.3.0"
fastapi = "^0.104.1"
uvicorn = {extras = ["standard"], version = "^0.24.0"}
pydantic = "^2.0.0"
```

### Step 2: Create Base Agent Structure

```python
# server/a2a_agents/base.py
from a2a import Agent, AgentError
from a2a.types import AgentCard, Skill, InputSchema
from typing import Dict, Any, Optional
import logging

class MedicalAgent(Agent):
    """Base class for medical domain agents using A2A SDK"""
    
    def __init__(self, name: str, description: str):
        super().__init__()
        self.logger = logging.getLogger(name)
        self._name = name
        self._description = description
    
    async def get_agent_card(self) -> AgentCard:
        """Return the agent's capabilities"""
        return AgentCard(
            name=self._name,
            description=self._description,
            skills=self._get_skills()
        )
    
    def _get_skills(self) -> list[Skill]:
        """Override in subclasses to define agent skills"""
        raise NotImplementedError
```

### Step 3: Implement Medical Agents

```python
# server/a2a_agents/medgemma_agent.py
from a2a import Agent
from a2a.types import AgentCard, Skill, InputSchema, OutputSchema
from a2a.decorators import skill
import httpx
from typing import Dict, Any

class MedGemmaAgent(Agent):
    """Medical Q&A agent using MedGemma model"""
    
    def __init__(self, llm_config):
        super().__init__()
        self.llm_config = llm_config
        self.http_client = httpx.AsyncClient()
    
    async def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="MedGemma Medical Assistant",
            description="Provides evidence-based medical information and answers",
            skills=[
                Skill(
                    name="answer_medical_question",
                    description="Answer general medical questions with clinical accuracy",
                    input_schema=InputSchema(
                        type="object",
                        properties={
                            "query": {"type": "string", "description": "Medical question"}
                        },
                        required=["query"]
                    ),
                    output_schema=OutputSchema(
                        type="object",
                        properties={
                            "answer": {"type": "string"},
                            "confidence": {"type": "number"},
                            "disclaimer": {"type": "string"}
                        }
                    )
                )
            ]
        )
    
    @skill("answer_medical_question")
    async def answer_medical_question(self, query: str) -> Dict[str, Any]:
        """Process medical question using MedGemma"""
        
        # Call LLM
        response = await self.http_client.post(
            f"{self.llm_config.base_url}/v1/chat/completions",
            json={
                "model": self.llm_config.med_model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a medical AI assistant. Provide accurate, "
                                 "evidence-based information. Always include appropriate disclaimers."
                    },
                    {"role": "user", "content": query}
                ],
                "temperature": 0.1,
                "max_tokens": 800
            },
            headers={"Authorization": f"Bearer {self.llm_config.api_key}"} if self.llm_config.api_key else {}
        )
        
        data = response.json()
        answer = data["choices"][0]["message"]["content"]
        
        return {
            "answer": answer,
            "confidence": 0.85,
            "disclaimer": "This information is for educational purposes only. "
                        "Please consult a healthcare professional for medical advice."
        }
```

### Step 4: Implement Clinical Research Agent

```python
# server/a2a_agents/clinical_agent.py
from a2a import Agent
from a2a.types import AgentCard, Skill, InputSchema
from a2a.decorators import skill
import httpx
from typing import Dict, Any, Optional

class ClinicalResearchAgent(Agent):
    """Agent for querying FHIR and clinical databases"""
    
    def __init__(self, llm_config, openmrs_config, spark_config):
        super().__init__()
        self.llm_config = llm_config
        self.openmrs_config = openmrs_config
        self.spark_config = spark_config
        self.http_client = httpx.AsyncClient()
    
    async def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="Clinical Research Assistant",
            description="Queries and synthesizes clinical data from FHIR and SQL sources",
            skills=[
                Skill(
                    name="clinical_research",
                    description="Retrieve and analyze clinical data",
                    input_schema=InputSchema(
                        type="object",
                        properties={
                            "query": {"type": "string"},
                            "scope": {
                                "type": "string",
                                "enum": ["facility", "hie"],
                                "default": "hie"
                            },
                            "facility_id": {"type": "string"},
                            "org_ids": {
                                "type": "array",
                                "items": {"type": "string"}
                            }
                        },
                        required=["query"]
                    )
                )
            ]
        )
    
    @skill("clinical_research")
    async def clinical_research(
        self,
        query: str,
        scope: str = "hie",
        facility_id: Optional[str] = None,
        org_ids: Optional[list] = None
    ) -> Dict[str, Any]:
        """Execute clinical research query"""
        
        # Determine data source based on query
        if "fhir" in query.lower() or "patient" in query.lower():
            data = await self._query_fhir(query, scope, facility_id, org_ids)
        else:
            data = await self._query_sql(query, scope, facility_id, org_ids)
        
        # Synthesize with MedGemma
        synthesis = await self._synthesize_response(query, data)
        
        return {
            "response": synthesis,
            "data_source": "FHIR" if "fhir" in query.lower() else "SQL",
            "scope": scope,
            "records_found": len(data) if isinstance(data, list) else 1
        }
    
    async def _query_fhir(self, query: str, scope: str, facility_id: str, org_ids: list):
        """Generate and execute FHIR query"""
        # Implementation here
        pass
    
    async def _query_sql(self, query: str, scope: str, facility_id: str, org_ids: list):
        """Generate and execute SQL query"""
        # Implementation here
        pass
    
    async def _synthesize_response(self, query: str, data: Any) -> str:
        """Use MedGemma to synthesize clinical response"""
        # Implementation here
        pass
```

### Step 5: Implement Router Agent

```python
# server/a2a_agents/router_agent.py
from a2a import Agent
from a2a.types import AgentCard, Skill, InputSchema
from a2a.decorators import skill
from a2a.client import AgentClient
import json
from typing import Dict, Any

class RouterAgent(Agent):
    """Orchestrator that routes requests to appropriate specialist agents"""
    
    def __init__(self, orchestrator_config, agent_urls: Dict[str, str]):
        super().__init__()
        self.orchestrator_config = orchestrator_config
        self.agent_urls = agent_urls
        self.clients = {
            name: AgentClient(url) 
            for name, url in agent_urls.items()
        }
    
    async def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="Medical Query Router",
            description="Routes medical queries to appropriate specialist agents",
            skills=[
                Skill(
                    name="route_query",
                    description="Analyze query and route to best agent",
                    input_schema=InputSchema(
                        type="object",
                        properties={
                            "query": {"type": "string"},
                            "context": {"type": "object"}
                        },
                        required=["query"]
                    )
                )
            ]
        )
    
    @skill("route_query")
    async def route_query(self, query: str, context: Dict = None) -> Dict[str, Any]:
        """Route query to appropriate specialist"""
        
        # Get available agents and their capabilities
        agent_cards = {}
        for name, client in self.clients.items():
            try:
                card = await client.get_agent_card()
                agent_cards[name] = card
            except Exception as e:
                self.logger.warning(f"Failed to get card for {name}: {e}")
        
        # Use LLM to select best agent
        selected_agent, selected_skill, args = await self._select_agent(
            query, agent_cards, context
        )
        
        # Invoke selected agent's skill
        client = self.clients[selected_agent]
        result = await client.invoke_skill(selected_skill, **args)
        
        return {
            "agent": selected_agent,
            "skill": selected_skill,
            "result": result
        }
    
    async def _select_agent(self, query: str, agent_cards: Dict, context: Dict):
        """Use orchestrator LLM to select best agent and skill"""
        
        # Format agent capabilities for LLM
        capabilities = []
        for name, card in agent_cards.items():
            for skill in card.skills:
                capabilities.append({
                    "agent": name,
                    "skill": skill.name,
                    "description": skill.description,
                    "input_schema": skill.input_schema
                })
        
        prompt = f"""
        Select the best agent and skill for this query:
        Query: {query}
        
        Available capabilities:
        {json.dumps(capabilities, indent=2)}
        
        Return JSON: {{"agent": str, "skill": str, "args": dict}}
        """
        
        # Call orchestrator LLM
        # ... LLM implementation
        
        return selected_agent, selected_skill, args
```

### Step 6: Create Agent Servers

```python
# server/a2a_agents/servers.py
from a2a.server import AgentServer
from fastapi import FastAPI
import uvicorn

def create_medgemma_server(config):
    """Create MedGemma agent server"""
    app = FastAPI(title="MedGemma Agent")
    agent = MedGemmaAgent(config.llm_config)
    server = AgentServer(agent)
    
    # Mount A2A endpoints
    app.mount("/", server.app)
    
    return app

def create_clinical_server(config):
    """Create Clinical Research agent server"""
    app = FastAPI(title="Clinical Research Agent")
    agent = ClinicalResearchAgent(
        config.llm_config,
        config.openmrs_config,
        config.spark_config
    )
    server = AgentServer(agent)
    app.mount("/", server.app)
    return app

def create_router_server(config, agent_urls):
    """Create Router agent server"""
    app = FastAPI(title="Router Agent")
    agent = RouterAgent(config.orchestrator_config, agent_urls)
    server = AgentServer(agent)
    app.mount("/", server.app)
    return app

# Launch scripts
if __name__ == "__main__":
    # Example: Launch MedGemma agent
    import sys
    from config import Config
    
    config = Config()
    
    if sys.argv[1] == "medgemma":
        app = create_medgemma_server(config)
        uvicorn.run(app, host="0.0.0.0", port=9101)
    elif sys.argv[1] == "clinical":
        app = create_clinical_server(config)
        uvicorn.run(app, host="0.0.0.0", port=9102)
    elif sys.argv[1] == "router":
        agent_urls = {
            "medgemma": "http://localhost:9101",
            "clinical": "http://localhost:9102"
        }
        app = create_router_server(config, agent_urls)
        uvicorn.run(app, host="0.0.0.0", port=9100)
```

### Step 7: Update FastAPI Bridge

```python
# server/main.py
from fastapi import FastAPI, HTTPException
from a2a.client import AgentClient
from pydantic import BaseModel
from typing import Optional

app = FastAPI(title="Medical Chat Bridge")

# A2A Router client
router_client = AgentClient("http://localhost:9100")

class ChatRequest(BaseModel):
    prompt: str
    conversation_id: Optional[str] = None
    scope: Optional[str] = "hie"
    facility_id: Optional[str] = None
    org_ids: Optional[list] = None

@app.post("/chat")
async def chat(request: ChatRequest):
    """Bridge endpoint that forwards to A2A router"""
    try:
        # Invoke router's route_query skill
        result = await router_client.invoke_skill(
            "route_query",
            query=request.prompt,
            context={
                "conversation_id": request.conversation_id,
                "scope": request.scope,
                "facility_id": request.facility_id,
                "org_ids": request.org_ids
            }
        )
        
        return {
            "response": result.get("result", {}).get("response", ""),
            "agent_used": result.get("agent"),
            "skill_used": result.get("skill")
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/agents")
async def list_agents():
    """List available agents via router"""
    card = await router_client.get_agent_card()
    return {"router": card.dict()}

@app.get("/health")
async def health():
    """Health check"""
    try:
        await router_client.get_agent_card()
        return {"status": "healthy", "a2a": "connected"}
    except:
        return {"status": "degraded", "a2a": "disconnected"}
```

## Deployment

### Using Docker Compose

```yaml
# docker-compose.yml
version: '3.8'

services:
  medgemma-agent:
    build: .
    command: python -m server.a2a_agents.servers medgemma
    ports:
      - "9101:9101"
    environment:
      - LLM_BASE_URL=${LLM_BASE_URL}
      - MED_MODEL=${MED_MODEL}
    networks:
      - a2a-network

  clinical-agent:
    build: .
    command: python -m server.a2a_agents.servers clinical
    ports:
      - "9102:9102"
    environment:
      - LLM_BASE_URL=${LLM_BASE_URL}
      - GENERAL_MODEL=${GENERAL_MODEL}
      - OPENMRS_FHIR_BASE_URL=${OPENMRS_FHIR_BASE_URL}
    networks:
      - a2a-network

  router-agent:
    build: .
    command: python -m server.a2a_agents.servers router
    ports:
      - "9100:9100"
    environment:
      - ORCHESTRATOR_PROVIDER=${ORCHESTRATOR_PROVIDER}
      - ORCHESTRATOR_MODEL=${ORCHESTRATOR_MODEL}
    depends_on:
      - medgemma-agent
      - clinical-agent
    networks:
      - a2a-network

  api-bridge:
    build: .
    command: uvicorn server.main:app --host 0.0.0.0 --port 3000
    ports:
      - "3000:3000"
    depends_on:
      - router-agent
    networks:
      - a2a-network

networks:
  a2a-network:
    driver: bridge
```

## Testing

### Test Agent Communication

```python
# tests/test_a2a_integration.py
import pytest
from a2a.client import AgentClient
import httpx

@pytest.mark.asyncio
async def test_medgemma_agent():
    """Test MedGemma agent"""
    client = AgentClient("http://localhost:9101")
    
    # Get agent card
    card = await client.get_agent_card()
    assert card.name == "MedGemma Medical Assistant"
    assert len(card.skills) > 0
    
    # Invoke skill
    result = await client.invoke_skill(
        "answer_medical_question",
        query="What are common symptoms of hypertension?"
    )
    assert "answer" in result
    assert "disclaimer" in result

@pytest.mark.asyncio
async def test_router_orchestration():
    """Test router orchestration"""
    client = AgentClient("http://localhost:9100")
    
    result = await client.invoke_skill(
        "route_query",
        query="What are the latest lab results for patient 123?"
    )
    
    assert result["agent"] == "clinical"  # Should route to clinical agent
    assert "result" in result
```

## Benefits of Using A2A SDK

1. **Standard Protocol**: JSON-RPC 2.0 ensures interoperability
2. **Agent Discovery**: Built-in agent card system
3. **Type Safety**: Pydantic models and schemas
4. **Error Handling**: Standard error codes and messages
5. **Streaming Support**: SSE for long-running operations
6. **Authentication**: Built-in auth mechanisms (when configured)
7. **Monitoring**: Standard metrics and tracing

## Migration Checklist

- [ ] Install A2A SDK (v0.3.0+)
- [ ] Remove custom a2a_layer.py
- [ ] Convert agents to SDK Agent class
- [ ] Implement @skill decorators
- [ ] Create AgentCard definitions
- [ ] Set up AgentServer for each agent
- [ ] Update FastAPI bridge to use AgentClient
- [ ] Test agent communication
- [ ] Update Docker deployment
- [ ] Document new endpoints

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Import errors | Ensure Python 3.10+ and `pip install a2a-sdk` |
| Agent not discoverable | Check AgentCard is properly defined |
| Skill invocation fails | Verify skill name matches @skill decorator |
| Network errors | Ensure all agents are on same Docker network |
| Type errors | Use Pydantic models for input/output validation |

## References

- [A2A Samples Repository](https://github.com/a2aproject/a2a-samples)
- [A2A Python SDK](https://github.com/a2aproject/a2a-python)
- [A2A Protocol Specification](https://a2aprotocol.ai)
- [JSON-RPC 2.0 Specification](https://www.jsonrpc.org/specification)
