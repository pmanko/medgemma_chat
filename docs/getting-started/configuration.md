# Configuration Guide

## Quick Start

```bash
cd projects/med-agent-hub
cp env.recommended .env
# Edit .env with your LM Studio IP address
```

## Essential Settings

### Network Configuration
```env
# CRITICAL: Set to your machine's IP for network access
AGENT_HOST_IP=127.0.0.1  # Local development
# AGENT_HOST_IP=10.0.0.41  # Network access
```

### LM Studio
```env
LLM_BASE_URL=http://10.0.0.41:1234  # Your LM Studio endpoint
```

### Model Assignments
```env
ORCHESTRATOR_MODEL=meta-llama-3.1-8b-instruct  # Router
MED_MODEL=medgemma-4b-it-mlx                   # Medical Q&A
CLINICAL_RESEARCH_MODEL=gemma-3-1b-it          # Clinical research
```

### Agent Ports
```env
A2A_ROUTER_PORT=9100
A2A_MEDGEMMA_PORT=9101
A2A_CLINICAL_PORT=9102
```

## Optional Features

### Google Gemini Orchestration
```env
ORCHESTRATOR_PROVIDER=gemini
GEMINI_API_KEY=your-api-key
ORCHESTRATOR_MODEL=gemini-1.5-flash
```

### FHIR Integration
```env
OPENMRS_FHIR_BASE_URL=http://localhost:8080/openmrs/ws/fhir2/R4/
OPENMRS_USERNAME=admin
OPENMRS_PASSWORD=Admin123
```

## Running the System

### All Services (Recommended)
```bash
honcho -f Procfile.dev start
```

### Individual Services
```bash
# Set environment file
export UVICORN_ENV_FILE=env.recommended

# Start each agent
poetry run uvicorn server.sdk_agents.router_server:app --port 9100
poetry run uvicorn server.sdk_agents.medgemma_server:app --port 9101
poetry run uvicorn server.sdk_agents.clinical_server:app --port 9102
```

### Testing
```bash
poetry run python test_models_direct.py
# Expected: "Test Summary: 3 out of 3 agents passed."
```

## Common Commands

### Check Agent Status
```bash
# Test agent cards
curl http://localhost:9100/.well-known/agent-card.json
curl http://localhost:9101/.well-known/agent-card.json
curl http://localhost:9102/.well-known/agent-card.json

# Test LM Studio
curl http://localhost:1234/v1/models
```

### View Logs
```bash
tail -f logs/router.log
tail -f logs/medgemma.log
tail -f logs/clinical.log
```

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Connection refused | Check `AGENT_HOST_IP` and LM Studio |
| Model not found | Verify model names in LM Studio |
| Agent timeout | Increase `CHAT_TIMEOUT_SECONDS` |
| messageId required | Add `messageId=str(uuid.uuid4())` |
| No response | Ensure `await updater.add_artifact()` called |

## Configuration Flow

```
env.recommended → server/config.py → Agent Executors
                ↓
         UVICORN_ENV_FILE 
         (via Procfile.dev)
```

The `server/config.py` file:
- Loads environment from `UVICORN_ENV_FILE` or `.env`
- Constructs agent URLs from `AGENT_HOST_IP` and ports
- Provides configuration objects to executors