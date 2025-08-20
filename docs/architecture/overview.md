# System Architecture Overview

The Medical Multi-Agent Chat System implements the A2A (Agent-to-Agent) protocol to enable collaborative medical AI through specialized, interoperable agents.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        User Interface                        │
│                    (Web Browser / API Client)                │
└───────────────────────┬─────────────────────────────────────┘
                        │ HTTP/HTTPS
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                      FastAPI Bridge                          │
│                  (Optional - for Web UI)                     │
└───────────────────────┬─────────────────────────────────────┘
                        │ A2A Protocol (JSON-RPC 2.0)
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                      Router Agent                            │
│                 (Orchestration Service)                      │
│                      Port: 9100                              │
└──────────┬──────────────────────────────┬───────────────────┘
           │                              │
           ▼                              ▼
┌──────────────────────┐      ┌──────────────────────────────┐
│   MedGemma Agent     │      │  Clinical Research Agent     │
│  (Medical Q&A)       │      │  (FHIR/SQL Queries)          │
│   Port: 9101         │      │      Port: 9102              │
└──────────┬───────────┘      └──────────┬───────────────────┘
           │                              │
           ▼                              ▼
┌──────────────────────────────────────────────────────────────┐
│                    External Services                         │
│         (LLM APIs, FHIR Servers, SQL Databases)              │
└──────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. User Interface Layer

The system provides multiple interfaces for interaction:

- **Web UI**: Browser-based chat interface using vanilla JavaScript
- **API Access**: Direct HTTP/JSON-RPC calls to agents
- **SDK Clients**: Python clients using the A2A SDK

### 2. FastAPI Bridge (Optional)

A lightweight translation layer that:
- Converts web requests to A2A protocol
- Manages conversation state
- Provides backwards compatibility
- Handles CORS and authentication

### 3. Agent Services

Three specialized microservices, each implementing the A2A protocol:

#### Router Agent (Port 9100)
- **Role**: Orchestrator and request router
- **Responsibilities**:
  - Discovers available agents via A2A protocol
  - Analyzes queries using LLM
  - Routes to appropriate specialist
  - Aggregates responses
- **Skills**: `route_query`

#### MedGemma Agent (Port 9101)
- **Role**: Medical knowledge expert
- **Responsibilities**:
  - Answers general medical questions
  - Provides evidence-based information
  - Includes appropriate disclaimers
- **Skills**: `answer_medical_question`

#### Clinical Research Agent (Port 9102)
- **Role**: Clinical data specialist
- **Responsibilities**:
  - Queries FHIR servers
  - Executes SQL on clinical databases
  - Synthesizes data into insights
- **Skills**: `clinical_research`

## A2A Protocol Implementation

### Protocol Standards

The system fully implements the A2A protocol:

```json
{
  "jsonrpc": "2.0",
  "method": "skill_name",
  "params": {
    "query": "user question",
    "additional": "parameters"
  },
  "id": 1
}
```

### Agent Discovery

Each agent exposes its capabilities via an agent card:

```json
{
  "name": "Agent Name",
  "description": "What this agent does",
  "version": "1.0.0",
  "skills": [
    {
      "name": "skill_name",
      "description": "What this skill does",
      "input_schema": { ... },
      "output_schema": { ... }
    }
  ]
}
```

### Message Flow

1. **Discovery Phase**:
   - Router queries each agent's `/.well-known/agent.json`
   - Builds catalog of available skills

2. **Routing Phase**:
   - User query arrives at router
   - LLM analyzes query and available skills
   - Selects best agent and skill

3. **Execution Phase**:
   - Router invokes selected skill via JSON-RPC
   - Agent processes request
   - Returns structured response

4. **Response Phase**:
   - Router receives agent response
   - Optionally aggregates multiple responses
   - Returns to user

## Technology Stack

### Core Technologies

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Language** | Python 3.10+ | Primary development language |
| **A2A SDK** | a2a-sdk 0.3.0+ | Protocol implementation |
| **Web Framework** | FastAPI | API bridge layer |
| **Async Runtime** | asyncio/uvicorn | Concurrent request handling |
| **HTTP Client** | httpx | Agent communication |
| **Validation** | Pydantic 2.0 | Type safety and validation |

### External Integrations

| Service | Purpose | Protocol |
|---------|---------|----------|
| **LM Studio** | Local LLM hosting | OpenAI API |
| **OpenAI** | Cloud LLM | REST API |
| **Google Gemini** | Advanced LLM | REST API |
| **OpenMRS** | FHIR server | FHIR R4 |
| **Spark Thrift** | SQL-on-FHIR | HiveServer2 |

## Design Principles

### 1. Microservices Architecture
- Each agent is independently deployable
- Services communicate via standard protocols
- No shared state between agents

### 2. Protocol-First Design
- Strict adherence to A2A specification
- JSON-RPC 2.0 for all agent communication
- Self-describing via agent cards

### 3. Async-First Implementation
- Non-blocking I/O throughout
- Concurrent request handling
- Efficient resource utilization

### 4. Type Safety
- Pydantic models for all data structures
- Runtime validation of inputs/outputs
- Clear error messages

### 5. Extensibility
- Easy to add new agents
- Skills can be added to existing agents
- Protocol allows for custom extensions

## Scalability Considerations

### Horizontal Scaling
- Agents can be replicated
- Load balancing via standard tools
- Stateless design enables scaling

### Performance Optimization
- Connection pooling for HTTP clients
- Response caching where appropriate
- Async processing for I/O operations

### Resource Management
- Configurable timeouts
- Memory limits per agent
- Graceful degradation

## Security Architecture

### Authentication & Authorization
- Optional API key authentication
- JWT support for user sessions
- Per-skill access control

### Data Protection
- TLS for agent communication
- Credential encryption
- Audit logging

### Medical Data Compliance
- FHIR standard compliance
- Scope-based access control
- Patient data anonymization options

## Deployment Patterns

### Development
```
All agents on localhost
Direct LLM connections
Mock FHIR data
```

### Staging
```
Docker Compose deployment
Shared LLM server
Test FHIR server
```

### Production
```
Kubernetes orchestration
Load-balanced agents
Production FHIR/databases
Monitoring and alerting
```

## Error Handling

### Graceful Degradation
- Fallback to alternative agents
- Cached responses for common queries
- Clear error messages to users

### Retry Logic
- Configurable retry attempts
- Exponential backoff
- Circuit breaker pattern

### Logging & Monitoring
- Structured logging (JSON)
- Distributed tracing support
- Prometheus metrics

## Future Architecture Enhancements

### Planned Improvements
1. **Vector Database Integration**: For RAG capabilities
2. **Event Streaming**: Kafka/RabbitMQ for async processing
3. **Multi-Region Deployment**: Geographic distribution
4. **Agent Marketplace**: Dynamic agent discovery and loading

### Extensibility Points
- Custom skill plugins
- Alternative LLM providers
- Additional data sources
- New agent types

## Summary

The architecture provides:
- ✅ **Standards-based** communication via A2A protocol
- ✅ **Modular design** with specialized agents
- ✅ **Scalable deployment** options
- ✅ **Extensible framework** for new capabilities
- ✅ **Production-ready** error handling and monitoring

This design ensures the system can grow from a local development setup to a production-scale medical AI platform while maintaining code quality and operational excellence.
