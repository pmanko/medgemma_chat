# System Architecture Overview

The Medical Multi-Agent Chat System implements the A2A (Agent-to-Agent) protocol using the A2A SDK v0.3.2+.

## Architecture

```
┌─────────────────────────────────────────────┐
│              Client/Browser                  │
└────────────────┬────────────────────────────┘
                 │ HTTP/JSON-RPC
                 ▼
┌─────────────────────────────────────────────┐
│         Router Agent (:9100)                 │
│         Orchestration & Routing              │
└──────┬─────────────────────┬─────────────────┘
       │                     │
       ▼                     ▼
┌──────────────┐      ┌──────────────────────┐
│  MedGemma    │      │  Clinical Research   │
│   (:9101)    │      │      (:9102)         │
└──────┬───────┘      └──────────┬───────────┘
       │                         │
       └────────┬────────────────┘
                ▼
         ┌──────────────┐
         │  LM Studio   │
         │   (:1234)    │
         └──────────────┘
```

## Core Components

### 1. Router Agent (Port 9100)
- Analyzes queries using LLM
- Routes to appropriate specialist
- Returns aggregated responses

### 2. MedGemma Agent (Port 9101)  
- Answers medical questions
- Provides evidence-based information
- Includes medical disclaimers

### 3. Clinical Research Agent (Port 9102)
- Handles clinical research questions
- Provides statistical analysis insights
- Optional FHIR data integration

## A2A Protocol Implementation

### Agent Discovery
Each agent exposes capabilities at `/.well-known/agent-card.json`:
- Name, description, version
- Available skills and their descriptions
- Transport preferences (JSON-RPC)

### Task Management
The SDK handles task lifecycle:
1. Create task from incoming message
2. Update status (working/completed/failed)
3. Add response artifacts
4. Stream updates via SSE

### Message Flow
```
Client → Router: JSON-RPC request
Router → LLM: Analyze query
Router → Agent: Forward to specialist
Agent → Router: Return artifacts
Router → Client: Final response
```

## Implementation Pattern

Each agent consists of:

1. **Executor** (`server/sdk_agents/*_executor.py`)
   - Inherits from `AgentExecutor`
   - Implements `execute()` method
   - Uses `TaskUpdater` for state management

2. **Server** (`server/sdk_agents/*_server.py`)
   - Creates `A2AStarletteApplication`
   - Loads agent card from JSON
   - Configures request handler

3. **Agent Card** (`server/agent_cards/*.json`)
   - Defines capabilities
   - Lists available skills

Example structure:
```python
# Executor
class MyExecutor(AgentExecutor):
    async def execute(self, context, event_queue):
        query = context.get_user_input()
        # Process with LLM
        # Update task status
        # Add artifacts
        # Complete task

# Server  
def create_server():
    executor = MyExecutor()
    handler = DefaultRequestHandler(executor, InMemoryTaskStore())
    server = A2AStarletteApplication(agent_card, handler)
    return server.build()
```

## Configuration

Environment variables control the system:
- `AGENT_HOST_IP`: Network IP for agents
- `LLM_BASE_URL`: LM Studio endpoint
- Model assignments per agent
- Port configurations

See [Configuration Guide](../getting-started/configuration.md) for details.

## Key Design Decisions

- **Pure A2A SDK**: No vendor dependencies
- **JSON Agent Cards**: Transparent capability definition
- **LLM Routing**: Intelligent query routing
- **Centralized Config**: Single source of truth
- **Process Management**: Honcho for development