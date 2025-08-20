# Configuration Guide

Simple configuration for the Medical Multi-Agent Chat System. Most users only need to set 2-3 variables.

## Quick Start

1. **Copy the example file**:
```bash
cp env.example .env
```

2. **Edit `.env`** with your LM Studio URL:
```env
LLM_BASE_URL=http://localhost:1234
```

3. **That's it!** The system uses smart defaults for everything else.

## Configuration Options

### Required Settings

Only these settings are required:

```env
# Your LM Studio endpoint (without /v1)
LLM_BASE_URL=http://localhost:1234

# Models available in your LM Studio
GENERAL_MODEL=llama-3-8b-instruct
MED_MODEL=medgemma-2  # Can be same as GENERAL_MODEL
```

### Optional: Use Google Gemini for Orchestration

Instead of using your local LLM for routing decisions, you can use Google Gemini for potentially better orchestration:

```env
# Uncomment these lines in your .env file
ORCHESTRATOR_PROVIDER=gemini
GEMINI_API_KEY=your-gemini-api-key-here
ORCHESTRATOR_MODEL=gemini-1.5-flash
```

**When to use Gemini orchestration:**
- You want faster routing decisions
- You have limited local compute
- You want to compare orchestration quality

**Getting a Gemini API key:**
1. Visit [Google AI Studio](https://aistudio.google.com/apikey)
2. Create a new API key
3. Copy it to your `.env` file

### Optional: Clinical Data Sources

If you have clinical data to query:

#### FHIR Server
```env
OPENMRS_FHIR_BASE_URL=http://localhost:8080/openmrs/ws/fhir2/R4/
OPENMRS_USERNAME=admin
OPENMRS_PASSWORD=Admin123
```

#### Local FHIR Parquet Files
```env
FHIR_PARQUET_DIR=/path/to/fhir/parquet/files
```

### Agent Ports

The default ports work for most setups. Only change if you have conflicts:

```env
A2A_ROUTER_URL=http://localhost:9100    # Router agent
A2A_MEDGEMMA_URL=http://localhost:9101  # Medical Q&A agent  
A2A_CLINICAL_URL=http://localhost:9102  # Clinical data agent
```

## Smart Defaults

The system automatically configures these settings (no need to set them):

| Setting | Default | Purpose |
|---------|---------|---------|
| `LLM_TEMPERATURE` | 0.2 | Lower = more consistent responses |
| `CHAT_TIMEOUT_SECONDS` | 90 | Maximum time for a chat response |
| `LOG_LEVEL` | INFO | Logging verbosity |
| `CORS_ORIGINS` | localhost:3000,8080 | Allowed web origins |
| `USE_HTTPS` | false | SSL for local dev not needed |

## Configuration Examples

### Minimal Setup (Most Users)
```env
# Just these two lines!
LLM_BASE_URL=http://localhost:1234
GENERAL_MODEL=llama-3-8b-instruct
```

### With Gemini Orchestration
```env
LLM_BASE_URL=http://localhost:1234
GENERAL_MODEL=llama-3-8b-instruct
MED_MODEL=medgemma-2

# Use Gemini for smarter routing
ORCHESTRATOR_PROVIDER=gemini
GEMINI_API_KEY=AIza...your-key-here
ORCHESTRATOR_MODEL=gemini-1.5-flash
```

### With Clinical Data
```env
LLM_BASE_URL=http://localhost:1234
GENERAL_MODEL=llama-3-8b-instruct

# Connect to FHIR server
OPENMRS_FHIR_BASE_URL=http://localhost:8080/openmrs/ws/fhir2/R4/
OPENMRS_USERNAME=admin
OPENMRS_PASSWORD=Admin123
```

## Troubleshooting

### Common Issues

| Problem | Solution |
|---------|----------|
| **"LLM_BASE_URL is required"** | Make sure `.env` file exists and contains `LLM_BASE_URL` |
| **"Connection refused"** | Check LM Studio is running on the correct port |
| **"Model not found"** | Verify model name matches what's loaded in LM Studio |
| **Gemini not working** | Check API key is valid and has necessary permissions |

### Checking Your Configuration

The system validates configuration on startup and shows a summary:

```
Configuration loaded:
  - LLM: http://localhost:1234 using llama-3-8b-instruct
  - Medical Model: medgemma-2
  - Orchestrator: openai using llama-3-8b-instruct
  - Environment: development
```

### Debug Mode

For troubleshooting, you can enable debug logging:

```env
LOG_LEVEL=DEBUG
```

## Advanced Configuration

For production deployments or advanced use cases, see the [Advanced Configuration Guide](../deployment/advanced-config.md).

## Running Commands

All commands should be run with Poetry:
```bash
# Launch agents
poetry run python launch_a2a_agents.py

# Test configuration
poetry run python test_config.py

# Run tests
poetry run python test_a2a_sdk.py
```

## Next Steps

✅ Configuration done? Continue with:
- [Quick Start](quick-start.md) - Run your first query
- [LM Studio Setup](lm-studio.md) - Configure your local LLM
- [Architecture Overview](../architecture/overview.md) - Understand the system