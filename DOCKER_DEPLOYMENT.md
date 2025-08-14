# Docker Deployment Guide for Multi-Agent Chat

## Overview

This guide explains how to build and deploy the Multi-Agent Chat application using Docker and the Instant OpenHIE v2 framework.

## Architecture

The application consists of two Docker containers:
- **Server Container**: FastAPI backend with A2A protocol support (Python 3.11 with Poetry)
- **Client Container**: nginx serving the static web UI with API proxy

## Prerequisites

1. Docker and Docker Compose installed
2. Docker Swarm initialized (for production deployment)
3. The `instant` Docker network created:
   ```bash
   docker network create instant
   ```
4. `jq` installed for JSON parsing in build scripts

## Quick Start

### 1. Configure Environment

Copy the environment template and configure your settings:

```bash
cd projects/multiagent_chat
cp env.template .env
# Edit .env with your configuration
```

Key configurations:
- `LLM_BASE_URL`: Your LLM endpoint (e.g., LM Studio at `http://localhost:1234`)
- `GENERAL_MODEL` and `MED_MODEL`: Model names to use
- `ORCHESTRATOR_PROVIDER`: Either `openai` or `gemini`
- OpenMRS and Spark settings if using FHIR integration

### 2. Build Docker Images

From the repository root:

```bash
# Build multiagent_chat images (server and client)
./build-custom-images.sh multiagent_chat
```

This will create:
- `multiagent-chat-server:latest`
- `multiagent-chat-client:latest`

### 3. Deploy with Docker Compose (Development)

For local development and testing:

```bash
cd packages/multiagent_chat
docker-compose up -d
```

Access the application:
- Client UI: http://localhost:8080
- Server API: http://localhost:3000

### 4. Deploy with Docker Swarm (Production)

For production deployment using Instant OpenHIE v2:

```bash
cd packages/multiagent_chat
./swarm.sh init
```

Manage the deployment:
```bash
./swarm.sh up     # Start services
./swarm.sh down   # Stop services
./swarm.sh destroy # Remove services and cleanup
```

## Container Details

### Server Container

- **Base Image**: Python 3.11-slim
- **Build Process**: Multi-stage build with Poetry
- **Port**: 3000
- **Health Check**: `/health` endpoint
- **Memory**: 1-2GB recommended (for ML models)
- **User**: Runs as non-root user `appuser`

### Client Container  

- **Base Image**: nginx:alpine
- **Port**: 80 (mapped to 8080 on host)
- **Features**: 
  - Serves static HTML/CSS/JS files
  - Proxies API calls to backend
  - CORS headers configured
  - Compression enabled
- **Memory**: 128-256MB

## nginx Proxy Configuration

The client container's nginx configuration handles:
- `/api/*` → Proxies to `http://multiagent-chat-server:3000/*`
- Direct endpoints (`/chat`, `/generate`, `/manifest`, `/health`) → Proxied to server
- Static files → Served with caching headers
- WebSocket support for future features

## Environment Variables

### Essential Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `LLM_BASE_URL` | OpenAI-compatible API endpoint | `http://localhost:1234` |
| `GENERAL_MODEL` | General purpose model name | `llama-3-8b-instruct` |
| `MED_MODEL` | Medical model name | `medgemma-2` |
| `ENABLE_A2A` | Enable A2A mode | `true` |

### Optional Integrations

| Variable | Description | Default |
|----------|-------------|---------|
| `OPENMRS_FHIR_BASE_URL` | OpenMRS FHIR endpoint | Empty (disabled) |
| `SPARK_THRIFT_HOST` | Spark server for SQL-on-FHIR | Empty (disabled) |
| `FHIR_PARQUET_DIR` | Local FHIR Parquet directory | Empty (disabled) |

## Troubleshooting

### Build Issues

1. **Poetry lock file missing**: Ensure `poetry.lock` exists in the project directory
2. **Build fails**: Check Docker daemon is running and has sufficient disk space
3. **jq not found**: Install jq package (`apt-get install jq` or `brew install jq`)

### Runtime Issues

1. **Client can't reach server**: Ensure both containers are on the `instant` network
2. **LLM timeout**: Increase `CHAT_TIMEOUT_SECONDS` and nginx proxy timeouts
3. **Memory issues**: Increase Docker memory limits in docker-compose.yml

### Debugging

View logs:
```bash
# Docker Swarm
docker service logs -f multiagent-chat_multiagent-chat-server
docker service logs -f multiagent-chat_multiagent-chat-client
```


## Integration with Instant OpenHIE

This package follows Instant OpenHIE v2 conventions:
- Uses the shared `instant` network
- Configured via `package-metadata.json`
- Managed through `swarm.sh` lifecycle scripts
- Environment variables loaded from package metadata

To integrate with other OpenHIE services, ensure they're on the same `instant` network and configure the appropriate URLs in your environment.

## Development Tips

1. **Hot Reload**: For development, mount the source code as volumes:
   ```yaml
   volumes:
     - ./server:/app/server
     - ./client:/usr/share/nginx/html
   ```

2. **Local LLM**: Use LM Studio or Ollama running on host:
   - Set `LLM_BASE_URL=http://host.docker.internal:1234` (Mac/Windows)
   - Set `LLM_BASE_URL=http://172.17.0.1:1234` (Linux)

3. **Custom Models**: Update model names in environment variables without rebuilding

## Security Considerations

- Both containers run as non-root users
- Secrets should be managed through Docker secrets in production
- Enable HTTPS termination at the reverse proxy level
- Consider API authentication for production deployments

## Next Steps

1. Configure your LLM backend (LM Studio, Ollama, or cloud API)
2. Set up OpenMRS FHIR integration if needed
3. Deploy and test the application
4. Monitor logs and health endpoints
5. Scale services as needed using Docker Swarm

For more details on the application itself, see the main README.md.
