# Agent Capabilities

Detailed documentation of each agent in the Medical Multi-Agent Chat System, their skills, and how they work together.

## Agent Overview

The system consists of three specialized agents, each with distinct responsibilities:

| Agent | Port | Primary Role | Key Skills |
|-------|------|--------------|------------|
| **Router** | 9100 | Orchestration & Routing | `route_query` |
| **MedGemma** | 9101 | Medical Q&A | `answer_medical_question` |
| **Clinical Research** | 9102 | Data Retrieval & Analysis | `clinical_research` |

## Router Agent

### Purpose
The Router Agent acts as the system's orchestrator, intelligently routing queries to the most appropriate specialist agent.

### Agent Card
```json
{
  "name": "Medical Query Router",
  "description": "Routes medical queries to appropriate specialist agents",
  "version": "1.0.0",
  "skills": [
    {
      "name": "route_query",
      "description": "Analyze query and route to best specialist agent"
    }
  ]
}
```

### Skills

#### `route_query`
Routes incoming queries to the most appropriate specialist agent.

**Input Schema**:
```json
{
  "type": "object",
  "properties": {
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
      "default": "hie"
    },
    "facility_id": {
      "type": "string",
      "description": "Facility identifier"
    },
    "org_ids": {
      "type": "array",
      "items": {"type": "string"}
    }
  },
  "required": ["query"]
}
```

**Output Schema**:
```json
{
  "type": "object",
  "properties": {
    "response": {
      "type": "string",
      "description": "Final response from specialist"
    },
    "agent_used": {
      "type": "string",
      "description": "Which agent handled the query"
    },
    "skill_used": {
      "type": "string",
      "description": "Which skill was invoked"
    },
    "routing_confidence": {
      "type": "number",
      "description": "Confidence in routing decision (0-1)"
    }
  }
}
```

### Routing Logic

The router uses an LLM to analyze:
1. Query content and intent
2. Available agents and their capabilities
3. Context (scope, facility, etc.)

**Routing Rules**:
- Medical questions → MedGemma Agent
- Data queries → Clinical Research Agent
- Complex queries → Multiple agents (future)

### Example Usage

```python
from a2a.client import AgentClient

router = AgentClient("http://localhost:9100")
result = await router.invoke_skill(
    "route_query",
    query="What are the symptoms of diabetes?",
    conversation_id="conv-123",
    scope="hie"
)
```

## MedGemma Agent

### Purpose
The MedGemma Agent specializes in answering medical questions using Google's MedGemma model, providing accurate, evidence-based medical information.

### Agent Card
```json
{
  "name": "MedGemma Medical Assistant",
  "description": "Provides evidence-based medical information and answers",
  "version": "1.0.0",
  "skills": [
    {
      "name": "answer_medical_question",
      "description": "Answer general medical questions with clinical accuracy"
    }
  ]
}
```

### Skills

#### `answer_medical_question`
Provides comprehensive answers to medical questions.

**Input Schema**:
```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Medical question to answer"
    },
    "include_references": {
      "type": "boolean",
      "description": "Whether to include medical references",
      "default": false
    }
  },
  "required": ["query"]
}
```

**Output Schema**:
```json
{
  "type": "object",
  "properties": {
    "answer": {
      "type": "string",
      "description": "Medical answer"
    },
    "confidence": {
      "type": "number",
      "description": "Confidence score (0-1)"
    },
    "disclaimer": {
      "type": "string",
      "description": "Medical disclaimer"
    },
    "references": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Medical references if requested"
    }
  }
}
```

### Capabilities

**Medical Domains**:
- General medicine
- Symptoms and diagnoses
- Treatments and medications
- Side effects and interactions
- Preventive care
- Medical procedures

**Response Features**:
- Evidence-based information
- Automatic disclaimers
- Confidence scoring
- Optional references
- Layman-friendly explanations

### Example Usage

```python
from a2a.client import AgentClient

medgemma = AgentClient("http://localhost:9101")
result = await medgemma.invoke_skill(
    "answer_medical_question",
    query="What are the side effects of metformin?",
    include_references=True
)
```

### Sample Responses

**Query**: "What are the symptoms of hypertension?"

**Response**:
```json
{
  "answer": "Hypertension, or high blood pressure, is often called the 'silent killer' because it typically has no symptoms in its early stages. However, when symptoms do occur, they may include:\n\n1. Headaches, particularly in the morning\n2. Dizziness or lightheadedness\n3. Blurred vision\n4. Nosebleeds\n5. Shortness of breath\n6. Chest pain\n7. Fatigue\n\nIt's important to note that these symptoms usually only appear when blood pressure reaches dangerously high levels...",
  "confidence": 0.92,
  "disclaimer": "This information is for educational purposes only...",
  "references": ["ACC/AHA Guidelines", "WHO Hypertension Guidelines"]
}
```

## Clinical Research Agent

### Purpose
The Clinical Research Agent specializes in querying and analyzing clinical data from FHIR servers and SQL databases.

### Agent Card
```json
{
  "name": "Clinical Research Assistant",
  "description": "Queries and synthesizes clinical data from FHIR and SQL sources",
  "version": "1.0.0",
  "skills": [
    {
      "name": "clinical_research",
      "description": "Retrieve and analyze clinical data with scope-based access"
    }
  ]
}
```

### Skills

#### `clinical_research`
Retrieves and analyzes clinical data from various sources.

**Input Schema**:
```json
{
  "type": "object",
  "properties": {
    "query": {
      "type": "string",
      "description": "Clinical research question or data request"
    },
    "scope": {
      "type": "string",
      "enum": ["facility", "hie"],
      "default": "hie",
      "description": "Access scope"
    },
    "facility_id": {
      "type": "string",
      "description": "Facility identifier for facility-scoped queries"
    },
    "org_ids": {
      "type": "array",
      "items": {"type": "string"},
      "description": "Organization identifiers"
    },
    "data_source": {
      "type": "string",
      "enum": ["auto", "fhir", "sql"],
      "default": "auto",
      "description": "Preferred data source"
    }
  },
  "required": ["query"]
}
```

**Output Schema**:
```json
{
  "type": "object",
  "properties": {
    "response": {
      "type": "string",
      "description": "Synthesized clinical response"
    },
    "data_source": {
      "type": "string",
      "description": "Data source used (FHIR/SQL)"
    },
    "scope": {
      "type": "string",
      "description": "Access scope applied"
    },
    "records_found": {
      "type": "integer",
      "description": "Number of records found"
    },
    "query_executed": {
      "type": "string",
      "description": "The actual query executed"
    }
  }
}
```

### Data Sources

**FHIR Servers**:
- OpenMRS FHIR R4
- HAPI FHIR
- Any FHIR R4 compliant server

**SQL Databases**:
- Spark SQL on Parquet
- DuckDB for local files
- Any JDBC-compatible database

### Query Generation

The agent uses LLMs to:
1. Convert natural language to FHIR paths
2. Generate SQL queries for analytics
3. Synthesize results into insights

### Example Usage

```python
from a2a.client import AgentClient

clinical = AgentClient("http://localhost:9102")
result = await clinical.invoke_skill(
    "clinical_research",
    query="Show blood pressure readings for diabetic patients",
    scope="facility",
    facility_id="F001",
    data_source="fhir"
)
```

### Sample Queries

**FHIR Query Example**:
- Input: "Recent lab results for patient 12345"
- Generated: `/Observation?patient=12345&category=laboratory&_sort=-date&_count=10`

**SQL Query Example**:
- Input: "Average blood pressure by age group"
- Generated: 
```sql
SELECT 
  FLOOR(age/10)*10 as age_group,
  AVG(systolic) as avg_systolic,
  AVG(diastolic) as avg_diastolic
FROM observations o
JOIN patients p ON o.patient_id = p.id
WHERE o.code = '85354-9'
GROUP BY FLOOR(age/10)*10
```

## Agent Collaboration Patterns

### Simple Routing
```
User → Router → MedGemma → Response
```
Example: "What is diabetes?"

### Data Query
```
User → Router → Clinical → FHIR/SQL → Response
```
Example: "Show recent lab results"

### Complex Query (Future)
```
User → Router → [MedGemma, Clinical] → Aggregated Response
```
Example: "Explain hypertension and show patients with it"

## Adding New Agents

### Agent Template

```python
from a2a import Agent
from a2a.types import AgentCard, Skill
from a2a.decorators import skill

class NewSpecialistAgent(Agent):
    
    async def get_agent_card(self) -> AgentCard:
        return AgentCard(
            name="New Specialist",
            description="What this agent does",
            version="1.0.0",
            skills=[
                Skill(
                    name="new_skill",
                    description="Skill description",
                    input_schema={...},
                    output_schema={...}
                )
            ]
        )
    
    @skill("new_skill")
    async def new_skill(self, param: str) -> dict:
        # Implementation
        return {"result": "..."}
```

### Registration Process

1. Create agent class
2. Define skills with schemas
3. Implement skill methods
4. Launch on unique port
5. Update router configuration

## Agent Configuration

### Environment Variables

Each agent can be configured via environment:

```env
# MedGemma Agent
MEDGEMMA_PORT=9101
MEDGEMMA_MAX_TOKENS=1000
MEDGEMMA_MODEL=medgemma-2

# Clinical Agent
CLINICAL_PORT=9102
CLINICAL_DEFAULT_SCOPE=hie
CLINICAL_MAX_RECORDS=100

# Router Agent
ROUTER_PORT=9100
ROUTER_CONFIDENCE_THRESHOLD=0.7
ROUTER_FALLBACK_AGENT=medgemma
```

## Monitoring Agents

### Health Checks

Each agent exposes health endpoints:

```bash
# Check agent status
curl http://localhost:9101/health

# Get agent metrics
curl http://localhost:9101/metrics
```

### Logging

Agents log key events:
- Skill invocations
- Query processing
- Error conditions
- Performance metrics

## Best Practices

1. **Single Responsibility**: Each agent should focus on one domain
2. **Clear Skills**: Well-defined inputs and outputs
3. **Error Handling**: Graceful degradation
4. **Documentation**: Comprehensive agent cards
5. **Testing**: Unit tests for each skill
6. **Monitoring**: Health checks and metrics

## Summary

The three-agent system provides:
- **Intelligent routing** via the Router Agent
- **Medical expertise** via the MedGemma Agent
- **Data analytics** via the Clinical Research Agent

This modular design allows for:
- Easy addition of new specialists
- Independent scaling of agents
- Clear separation of concerns
- Robust error handling
- Future extensibility
