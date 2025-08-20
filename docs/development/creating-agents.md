# Creating New Agents

Learn how to extend the Medical Multi-Agent Chat System by creating your own specialized A2A-compliant agents.

## Overview

Creating a new agent involves:
1. Defining the agent's purpose and skills
2. Implementing the A2A Agent interface
3. Defining input/output schemas
4. Writing skill implementation
5. Testing and deploying

## Basic Agent Structure

### Minimal Agent Template

```python
# my_agent.py
from a2a import Agent
from a2a.types import AgentCard, Skill, InputSchema, OutputSchema
from a2a.decorators import skill
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)

class MySpecialistAgent(Agent):
    """Your specialized agent description"""
    
    def __init__(self):
        super().__init__()
        # Initialize any resources (LLM clients, DB connections, etc.)
        logger.info("MySpecialistAgent initialized")
    
    async def get_agent_card(self) -> AgentCard:
        """Define agent capabilities"""
        return AgentCard(
            name="My Specialist Agent",
            description="What this agent specializes in",
            version="1.0.0",
            skills=[
                Skill(
                    name="my_skill",
                    description="What this skill does",
                    input_schema=InputSchema(
                        type="object",
                        properties={
                            "query": {"type": "string", "description": "Input query"}
                        },
                        required=["query"]
                    ),
                    output_schema=OutputSchema(
                        type="object",
                        properties={
                            "result": {"type": "string", "description": "Result"}
                        }
                    )
                )
            ]
        )
    
    @skill("my_skill")
    async def my_skill(self, query: str) -> Dict[str, Any]:
        """Implement the skill logic"""
        # Your implementation here
        result = await self.process_query(query)
        return {"result": result}
    
    async def process_query(self, query: str) -> str:
        """Internal processing logic"""
        # Your processing logic
        return f"Processed: {query}"
    
    async def cleanup(self):
        """Clean up resources on shutdown"""
        logger.info("Cleaning up MySpecialistAgent")
```

## Real-World Example: Pharmacy Agent

Let's create a pharmacy agent that handles medication-related queries:

```python
# pharmacy_agent.py
from a2a import Agent
from a2a.types import AgentCard, Skill, InputSchema, OutputSchema
from a2a.decorators import skill
import httpx
import logging
from typing import Dict, Any, List, Optional
import os

logger = logging.getLogger(__name__)

class PharmacyAgent(Agent):
    """Agent specializing in medication and pharmacy queries"""
    
    def __init__(self):
        super().__init__()
        self.llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:1234")
        self.llm_api_key = os.getenv("LLM_API_KEY", "")
        self.model = os.getenv("PHARMACY_MODEL", "llama-3-8b-instruct")
        self.http_client = httpx.AsyncClient(timeout=60.0)
        
        # Load drug database or API credentials
        self.drug_api_url = os.getenv("DRUG_API_URL", "https://api.fda.gov/drug/")
        logger.info("PharmacyAgent initialized")
    
    async def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="Pharmacy Assistant",
            description="Handles medication queries, interactions, and pharmacy information",
            version="1.0.0",
            skills=[
                Skill(
                    name="check_drug_interaction",
                    description="Check for interactions between medications",
                    input_schema=InputSchema(
                        type="object",
                        properties={
                            "medications": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "List of medications to check"
                            },
                            "include_severity": {
                                "type": "boolean",
                                "default": True,
                                "description": "Include interaction severity"
                            }
                        },
                        required=["medications"]
                    ),
                    output_schema=OutputSchema(
                        type="object",
                        properties={
                            "interactions": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "drugs": {"type": "array", "items": {"type": "string"}},
                                        "description": {"type": "string"},
                                        "severity": {"type": "string"}
                                    }
                                }
                            },
                            "safe": {"type": "boolean"},
                            "warnings": {"type": "array", "items": {"type": "string"}}
                        }
                    )
                ),
                Skill(
                    name="medication_info",
                    description="Get detailed information about a medication",
                    input_schema=InputSchema(
                        type="object",
                        properties={
                            "medication": {"type": "string", "description": "Medication name"},
                            "info_type": {
                                "type": "string",
                                "enum": ["dosage", "side_effects", "usage", "all"],
                                "default": "all"
                            }
                        },
                        required=["medication"]
                    )
                )
            ]
        )
    
    @skill("check_drug_interaction")
    async def check_drug_interaction(
        self, 
        medications: List[str], 
        include_severity: bool = True
    ) -> Dict[str, Any]:
        """Check for drug interactions between medications"""
        
        if len(medications) < 2:
            return {
                "interactions": [],
                "safe": True,
                "warnings": ["Need at least 2 medications to check interactions"]
            }
        
        # Query drug interaction database or API
        interactions = await self._query_interactions(medications)
        
        # Analyze with LLM for detailed explanation
        prompt = f"""Analyze these drug interactions:
        Medications: {', '.join(medications)}
        Known interactions: {interactions}
        
        Provide a clinical assessment of safety and recommendations."""
        
        analysis = await self._call_llm(prompt)
        
        return {
            "interactions": interactions,
            "safe": len(interactions) == 0,
            "warnings": self._extract_warnings(analysis)
        }
    
    @skill("medication_info")
    async def medication_info(
        self, 
        medication: str, 
        info_type: str = "all"
    ) -> Dict[str, Any]:
        """Get detailed medication information"""
        
        # Query drug database
        drug_data = await self._query_drug_database(medication)
        
        if not drug_data:
            return {
                "error": f"Medication '{medication}' not found",
                "suggestions": await self._suggest_similar_drugs(medication)
            }
        
        # Format response based on requested info type
        if info_type == "dosage":
            return {"medication": medication, "dosage": drug_data.get("dosage", {})}
        elif info_type == "side_effects":
            return {"medication": medication, "side_effects": drug_data.get("side_effects", [])}
        elif info_type == "usage":
            return {"medication": medication, "usage": drug_data.get("usage", "")}
        else:  # all
            return drug_data
    
    async def _query_interactions(self, medications: List[str]) -> List[Dict]:
        """Query drug interaction database"""
        # This would connect to a real drug interaction API
        # For demo, using LLM knowledge
        prompt = f"""List known drug interactions between: {', '.join(medications)}
        Format as JSON array with: drugs (array), description, severity (major/moderate/minor)"""
        
        response = await self._call_llm(prompt)
        try:
            import json
            return json.loads(response)
        except:
            return []
    
    async def _query_drug_database(self, medication: str) -> Optional[Dict]:
        """Query drug information database"""
        # In production, this would query FDA API or drug database
        prompt = f"""Provide comprehensive information about {medication}:
        - Generic name
        - Brand names
        - Drug class
        - Common dosages
        - Side effects
        - Usage instructions
        - Warnings
        Format as JSON."""
        
        response = await self._call_llm(prompt)
        try:
            import json
            return json.loads(response)
        except:
            return None
    
    async def _call_llm(self, prompt: str) -> str:
        """Call LLM for analysis"""
        headers = {"Content-Type": "application/json"}
        if self.llm_api_key:
            headers["Authorization"] = f"Bearer {self.llm_api_key}"
        
        response = await self.http_client.post(
            f"{self.llm_base_url}/v1/chat/completions",
            json={
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are a pharmacy expert."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 1000
            },
            headers=headers
        )
        
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    
    def _extract_warnings(self, analysis: str) -> List[str]:
        """Extract warnings from analysis"""
        # Simple extraction logic
        warnings = []
        for line in analysis.split('\n'):
            if 'warning' in line.lower() or 'caution' in line.lower():
                warnings.append(line.strip())
        return warnings
    
    async def _suggest_similar_drugs(self, medication: str) -> List[str]:
        """Suggest similar medication names"""
        # In production, use fuzzy matching against drug database
        return []
    
    async def cleanup(self):
        """Clean up resources"""
        await self.http_client.aclose()
        logger.info("PharmacyAgent cleanup completed")
```

## Launching Your Agent

### Standalone Launch Script

```python
# launch_pharmacy.py
import asyncio
import uvicorn
from a2a.server import AgentServer
from pharmacy_agent import PharmacyAgent

async def main():
    # Create agent instance
    agent = PharmacyAgent()
    
    # Create A2A server
    server = AgentServer(agent)
    
    # Configure and run
    config = uvicorn.Config(
        server.app,
        host="0.0.0.0",
        port=9103,  # Choose unique port
        log_level="info"
    )
    
    server_instance = uvicorn.Server(config)
    await server_instance.serve()

if __name__ == "__main__":
    asyncio.run(main())
```

### Integration with Existing System

Update the router agent to know about your new agent:

```python
# In router_agent.py or environment config
self.agent_urls = {
    "medgemma": "http://localhost:9101",
    "clinical": "http://localhost:9102",
    "pharmacy": "http://localhost:9103"  # Add your agent
}
```

## Testing Your Agent

### Unit Tests

```python
# test_pharmacy_agent.py
import pytest
from a2a.client import AgentClient
from pharmacy_agent import PharmacyAgent

@pytest.fixture
async def agent():
    return PharmacyAgent()

@pytest.fixture
async def client():
    # Start agent server for testing
    return AgentClient("http://localhost:9103")

@pytest.mark.asyncio
async def test_agent_card(agent):
    card = await agent.get_agent_card()
    assert card.name == "Pharmacy Assistant"
    assert len(card.skills) == 2

@pytest.mark.asyncio
async def test_drug_interaction(agent):
    result = await agent.check_drug_interaction(
        medications=["aspirin", "warfarin"],
        include_severity=True
    )
    assert "interactions" in result
    assert "safe" in result
    assert isinstance(result["safe"], bool)

@pytest.mark.asyncio
async def test_medication_info(agent):
    result = await agent.medication_info(
        medication="metformin",
        info_type="dosage"
    )
    assert "medication" in result
    assert result["medication"] == "metformin"
```

### Integration Tests

```python
# test_integration.py
import asyncio
from a2a.client import AgentClient

async def test_pharmacy_integration():
    """Test pharmacy agent with router"""
    router = AgentClient("http://localhost:9100")
    
    # Router should route pharmacy queries to pharmacy agent
    result = await router.invoke_skill(
        "route_query",
        query="Check interaction between aspirin and ibuprofen"
    )
    
    assert result["agent_used"] == "pharmacy"
    assert "interactions" in result["response"]

asyncio.run(test_pharmacy_integration())
```

## Best Practices

### 1. Agent Design

- **Single Responsibility**: Each agent should focus on one domain
- **Clear Skills**: Well-defined, specific skills with clear inputs/outputs
- **Composability**: Design skills that can work with other agents

### 2. Schema Definition

```python
# Use detailed schemas for better interoperability
input_schema = InputSchema(
    type="object",
    properties={
        "query": {
            "type": "string",
            "description": "Detailed description helps other agents understand",
            "minLength": 1,
            "maxLength": 1000
        },
        "options": {
            "type": "object",
            "properties": {
                "format": {"type": "string", "enum": ["json", "text"]},
                "verbose": {"type": "boolean", "default": False}
            }
        }
    },
    required=["query"],
    additionalProperties=False
)
```

### 3. Error Handling

```python
@skill("my_skill")
async def my_skill(self, query: str) -> Dict[str, Any]:
    try:
        result = await self.process(query)
        return {"success": True, "result": result}
    except ValueError as e:
        return {"success": False, "error": str(e), "error_type": "validation"}
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return {"success": False, "error": "Internal error", "error_type": "internal"}
```

### 4. Resource Management

```python
class MyAgent(Agent):
    def __init__(self):
        super().__init__()
        self.resources = []
    
    async def initialize(self):
        """Initialize expensive resources"""
        self.db_conn = await create_connection()
        self.resources.append(self.db_conn)
    
    async def cleanup(self):
        """Clean up all resources"""
        for resource in self.resources:
            try:
                await resource.close()
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
```

### 5. Logging and Monitoring

```python
import time

@skill("timed_skill")
async def timed_skill(self, query: str) -> Dict[str, Any]:
    start_time = time.time()
    
    logger.info(f"Processing query: {query[:100]}...")
    
    try:
        result = await self.process(query)
        elapsed = time.time() - start_time
        
        logger.info(f"Query processed in {elapsed:.2f}s")
        
        return {
            "result": result,
            "processing_time": elapsed
        }
    except Exception as e:
        logger.error(f"Error processing query: {e}", exc_info=True)
        raise
```

## Advanced Topics

### Multi-Skill Agents

```python
class AdvancedAgent(Agent):
    
    @skill("analyze")
    async def analyze(self, data: str) -> Dict:
        """Analyze data"""
        analysis = await self._analyze(data)
        return {"analysis": analysis}
    
    @skill("summarize")
    async def summarize(self, text: str, max_length: int = 100) -> Dict:
        """Summarize text"""
        summary = await self._summarize(text, max_length)
        return {"summary": summary}
    
    @skill("compare")
    async def compare(self, item1: str, item2: str) -> Dict:
        """Compare two items"""
        comparison = await self._compare(item1, item2)
        return {"comparison": comparison}
```

### Stateful Agents

```python
class StatefulAgent(Agent):
    def __init__(self):
        super().__init__()
        self.sessions = {}  # Store session state
    
    @skill("start_session")
    async def start_session(self, user_id: str) -> Dict:
        session_id = str(uuid.uuid4())
        self.sessions[session_id] = {
            "user_id": user_id,
            "history": [],
            "created_at": datetime.now()
        }
        return {"session_id": session_id}
    
    @skill("continue_session")
    async def continue_session(self, session_id: str, query: str) -> Dict:
        if session_id not in self.sessions:
            return {"error": "Invalid session"}
        
        session = self.sessions[session_id]
        session["history"].append(query)
        
        # Process with context
        response = await self.process_with_history(query, session["history"])
        return {"response": response}
```

## Deployment Checklist

- [ ] Agent implements A2A Agent interface
- [ ] All skills have proper input/output schemas
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Unit tests written
- [ ] Integration tests pass
- [ ] Documentation updated
- [ ] Unique port assigned
- [ ] Router configuration updated
- [ ] Docker image built
- [ ] Health check endpoint works
- [ ] Monitoring configured

## Troubleshooting

| Issue | Solution |
|-------|----------|
| **Agent not discovered** | Check agent card is properly defined |
| **Skill not found** | Verify skill name matches @skill decorator |
| **Schema validation errors** | Ensure input matches defined schema |
| **Timeout errors** | Increase timeout, optimize processing |
| **Memory leaks** | Implement proper cleanup methods |

## Next Steps

- Add your agent to the [Router configuration](../architecture/agents.md#router-agent)
- Create [Docker image](../deployment/docker.md) for your agent
- Add [monitoring](../deployment/monitoring.md) for production
- Share your agent with the community!

