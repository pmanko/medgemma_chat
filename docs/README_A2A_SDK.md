# Multi-Agent Chat with Official A2A SDK

This implementation uses the official [A2A SDK](https://github.com/a2aproject/a2a-samples) to build a medical AI multi-agent system with proper protocol compliance.

## Quick Start

### 1. Install Dependencies

```bash
# Requires Python 3.10+
pip install a2a-sdk

# Or use poetry
poetry install
```

### 2. Configure Environment

```bash
cp env.example .env
# Edit .env with your LLM settings
```

### 3. Start Agents

**Option A: Start all agents together**
```bash
python launch_a2a_agents.py
```

**Option B: Start agents individually (for development)**
```bash
# Terminal 1: MedGemma Agent
python launch_a2a_agents.py medgemma

# Terminal 2: Clinical Research Agent  
python launch_a2a_agents.py clinical

# Terminal 3: Router Agent
python launch_a2a_agents.py router
```

### 4. Test the System

```bash
# Run comprehensive tests
python test_a2a_sdk.py
```

### 5. Use the Web UI

Open `client/index.html` in your browser and select "Agents (A2A)" mode.

## Architecture

```
┌─────────────────────────────────┐
│         Web Client              │
│    (HTML/JS - Pico CSS)         │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│      FastAPI Bridge             │
│    (Optional - for Web UI)      │
└─────────────┬───────────────────┘
              │
              ▼ A2A Protocol (JSON-RPC 2.0)
┌─────────────────────────────────┐
│      Router Agent               │
│    (Orchestration)              │
│    Port: 9100                   │
└──────┬──────────────┬───────────┘
       │              │
       ▼              ▼
┌──────────────┐ ┌────────────────┐
│  MedGemma    │ │   Clinical     │
│   Agent      │ │  Research      │
│ Port: 9101   │ │  Port: 9102    │
└──────────────┘ └────────────────┘
```

## Agent Capabilities

### MedGemma Agent
- **Skill**: `answer_medical_question`
- **Purpose**: General medical Q&A
- **Model**: MedGemma-2
- **Endpoint**: http://localhost:9101

### Clinical Research Agent
- **Skill**: `clinical_research`
- **Purpose**: Query FHIR data and clinical databases
- **Models**: Gemma (queries) + MedGemma (synthesis)
- **Endpoint**: http://localhost:9102
- **Data Sources**: OpenMRS FHIR, SQL-on-FHIR

### Router Agent
- **Skill**: `route_query`
- **Purpose**: Orchestrate between specialist agents
- **Model**: Configurable (Llama/Gemini)
- **Endpoint**: http://localhost:9100

## A2A Protocol Details

### Agent Discovery

Each agent exposes its capabilities via the standard A2A agent card:

```bash
curl -X POST http://localhost:9101 \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "get_agent_card",
    "params": {},
    "id": 1
  }'
```

Response:
```json
{
  "jsonrpc": "2.0",
  "result": {
    "name": "MedGemma Medical Assistant",
    "description": "Provides evidence-based medical information",
    "version": "1.0.0",
    "skills": [
      {
        "name": "answer_medical_question",
        "description": "Answer general medical questions",
        "input_schema": {...},
        "output_schema": {...}
      }
    ]
  },
  "id": 1
}
```

### Skill Invocation

Invoke agent skills using the A2A protocol:

```bash
curl -X POST http://localhost:9101 \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "method": "answer_medical_question",
    "params": {
      "query": "What are symptoms of diabetes?",
      "include_references": true
    },
    "id": 2
  }'
```

## Python Client Usage

```python
from a2a.client import AgentClient

# Connect to an agent
client = AgentClient("http://localhost:9101")

# Get agent capabilities
card = await client.get_agent_card()
print(f"Agent: {card.name}")
print(f"Skills: {[s.name for s in card.skills]}")

# Invoke a skill
result = await client.invoke_skill(
    "answer_medical_question",
    query="What causes hypertension?",
    include_references=True
)

print(f"Answer: {result['answer']}")
print(f"Confidence: {result['confidence']}")
```

## Development

### Creating New Agents

```python
from a2a import Agent
from a2a.types import AgentCard, Skill
from a2a.decorators import skill

class MyCustomAgent(Agent):
    
    async def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="My Custom Agent",
            description="Description",
            skills=[
                Skill(
                    name="my_skill",
                    description="What it does",
                    input_schema={...},
                    output_schema={...}
                )
            ]
        )
    
    @skill("my_skill")
    async def my_skill(self, param1: str) -> Dict:
        # Implementation
        return {"result": "..."}
```

### Testing Agents

```python
import pytest
from a2a.client import AgentClient

@pytest.mark.asyncio
async def test_agent():
    client = AgentClient("http://localhost:9101")
    
    # Test discovery
    card = await client.get_agent_card()
    assert card.name == "Expected Name"
    
    # Test skill
    result = await client.invoke_skill(
        "skill_name",
        param="value"
    )
    assert "expected_key" in result
```

## Environment Variables

```bash
# LLM Configuration
LLM_BASE_URL=http://localhost:1234
LLM_API_KEY=
GENERAL_MODEL=llama-3-8b-instruct
MED_MODEL=medgemma-2
LLM_TEMPERATURE=0.2

# Orchestrator
ORCHESTRATOR_PROVIDER=openai  # or gemini
ORCHESTRATOR_MODEL=llama-3-8b-instruct
GEMINI_API_KEY=  # if using gemini

# A2A Service URLs
A2A_ROUTER_URL=http://localhost:9100
A2A_MEDGEMMA_URL=http://localhost:9101
A2A_CLINICAL_URL=http://localhost:9102

# Data Sources (optional)
OPENMRS_FHIR_BASE_URL=
OPENMRS_USERNAME=
OPENMRS_PASSWORD=
SPARK_THRIFT_HOST=
SPARK_THRIFT_PORT=10000
SPARK_THRIFT_DATABASE=default
```

## Docker Deployment

```yaml
# docker-compose.a2a.yml
version: '3.8'

services:
  medgemma-agent:
    build: .
    command: python launch_a2a_agents.py medgemma
    ports:
      - "9101:9101"
    environment:
      - LLM_BASE_URL=${LLM_BASE_URL}
      - MED_MODEL=${MED_MODEL}

  clinical-agent:
    build: .
    command: python launch_a2a_agents.py clinical
    ports:
      - "9102:9102"
    environment:
      - LLM_BASE_URL=${LLM_BASE_URL}
      - GENERAL_MODEL=${GENERAL_MODEL}

  router-agent:
    build: .
    command: python launch_a2a_agents.py router
    ports:
      - "9100:9100"
    depends_on:
      - medgemma-agent
      - clinical-agent
    environment:
      - ORCHESTRATOR_PROVIDER=${ORCHESTRATOR_PROVIDER}
      - A2A_MEDGEMMA_URL=http://medgemma-agent:9101
      - A2A_CLINICAL_URL=http://clinical-agent:9102
```

## Monitoring

### Health Checks

Each agent exposes health status via A2A protocol:

```bash
# Check agent health
curl http://localhost:9101/health
```

### Logging

Agents log to stdout with structured logging:
```
2024-01-15 10:30:45 - RouterAgent - INFO - Routing query to medgemma.answer_medical_question
2024-01-15 10:30:46 - MedGemmaAgent - INFO - Processing medical query: What are symptoms...
```

## Benefits of A2A SDK

1. **Protocol Compliance**: Full JSON-RPC 2.0 implementation
2. **Type Safety**: Pydantic models for all inputs/outputs
3. **Discovery**: Standard agent card format
4. **Error Handling**: Consistent error codes and messages
5. **Streaming**: Built-in SSE support for long operations
6. **Testing**: Mock client for unit tests
7. **Documentation**: Auto-generated from schemas

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: a2a` | Install SDK: `pip install a2a-sdk` |
| Agent not responding | Check port is free: `lsof -i :9101` |
| Routing errors | Verify all agents are running |
| LLM timeout | Increase timeout in AgentClient |
| FHIR connection failed | Check OPENMRS_FHIR_BASE_URL |

## Next Steps

1. **Add more agents**: Imaging, Lab, Pharmacy specialists
2. **Implement authentication**: A2A auth extensions
3. **Add persistence**: Store conversations
4. **Deploy to cloud**: Kubernetes manifests
5. **Monitor performance**: Prometheus metrics

## References

- [A2A Protocol Specification](https://a2aprotocol.ai)
- [A2A SDK Documentation](https://github.com/a2aproject/a2a-python)
- [A2A Samples](https://github.com/a2aproject/a2a-samples)
- [JSON-RPC 2.0](https://www.jsonrpc.org/specification)
