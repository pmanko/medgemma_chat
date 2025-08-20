# Docker Deployment Guide

Deploy the Medical Multi-Agent Chat System using Docker for consistent, reproducible environments.

## Overview

The system uses Docker to containerize each A2A agent as a separate microservice, enabling easy deployment, scaling, and isolation.

## Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Router      │     │  MedGemma    │     │  Clinical    │
│  Agent       │────▶│  Agent       │     │  Agent       │
│  :9100       │     │  :9101       │     │  :9102       │
└──────────────┘     └──────────────┘     └──────────────┘
       ▲                                           │
       │                                           ▼
┌──────────────┐                          ┌──────────────┐
│   Web UI     │                          │   External   │
│   nginx      │                          │   Services   │
│   :8080      │                          │  (LLM, FHIR) │
└──────────────┘                          └──────────────┘
```

## Quick Start

### 1. Build and Run with Docker Compose

```bash
# Clone repository
git clone <repo>
cd projects/multiagent_chat

# Configure environment
cp env.example .env
# Edit .env with your LLM settings

# Build and start all services
docker-compose up --build

# Access the UI at http://localhost:8080
```

### 2. Production Deployment

```bash
# Build production images
docker-compose -f docker-compose.prod.yml build

# Deploy with Docker Swarm
docker stack deploy -c docker-compose.prod.yml multiagent

# Or use Kubernetes
kubectl apply -f k8s/
```

## Docker Images

### Base Agent Image

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY server/ ./server/
COPY launch_a2a_agents.py .

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Default command (override per agent)
CMD ["python", "launch_a2a_agents.py"]
```

### Agent-Specific Images

Each agent can have its own Dockerfile for customization:

```dockerfile
# Dockerfile.medgemma
FROM multiagent-base:latest

ENV AGENT_TYPE=medgemma
ENV PORT=9101

EXPOSE 9101

CMD ["python", "launch_a2a_agents.py", "medgemma"]
```

## Docker Compose Configuration

### Development Configuration

```yaml
# docker-compose.yml
version: '3.8'

services:
  router-agent:
    build: .
    command: python launch_a2a_agents.py router
    ports:
      - "9100:9100"
    environment:
      - LLM_BASE_URL=${LLM_BASE_URL}
      - ORCHESTRATOR_MODEL=${ORCHESTRATOR_MODEL}
      - A2A_MEDGEMMA_URL=http://medgemma-agent:9101
      - A2A_CLINICAL_URL=http://clinical-agent:9102
    networks:
      - multiagent-network
    depends_on:
      - medgemma-agent
      - clinical-agent
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9100/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  medgemma-agent:
    build: .
    command: python launch_a2a_agents.py medgemma
    ports:
      - "9101:9101"
    environment:
      - LLM_BASE_URL=${LLM_BASE_URL}
      - MED_MODEL=${MED_MODEL}
    networks:
      - multiagent-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9101/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  clinical-agent:
    build: .
    command: python launch_a2a_agents.py clinical
    ports:
      - "9102:9102"
    environment:
      - LLM_BASE_URL=${LLM_BASE_URL}
      - GENERAL_MODEL=${GENERAL_MODEL}
      - OPENMRS_FHIR_BASE_URL=${OPENMRS_FHIR_BASE_URL}
      - SPARK_THRIFT_HOST=${SPARK_THRIFT_HOST}
    networks:
      - multiagent-network
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:9102/health"]
      interval: 30s
      timeout: 10s
      retries: 3

  web-ui:
    image: nginx:alpine
    ports:
      - "8080:80"
    volumes:
      - ./client:/usr/share/nginx/html:ro
      - ./nginx.conf:/etc/nginx/conf.d/default.conf:ro
    networks:
      - multiagent-network
    depends_on:
      - router-agent

networks:
  multiagent-network:
    driver: bridge
```

### Production Configuration

```yaml
# docker-compose.prod.yml
version: '3.8'

services:
  router-agent:
    image: multiagent-router:${VERSION:-latest}
    deploy:
      replicas: 2
      resources:
        limits:
          cpus: '1.0'
          memory: 1G
        reservations:
          cpus: '0.5'
          memory: 512M
      restart_policy:
        condition: on-failure
        delay: 5s
        max_attempts: 3
    environment:
      - LLM_BASE_URL=${LLM_BASE_URL}
      - LOG_LEVEL=WARNING
    networks:
      - multiagent-network
    secrets:
      - llm_api_key

  medgemma-agent:
    image: multiagent-medgemma:${VERSION:-latest}
    deploy:
      replicas: 3
      resources:
        limits:
          cpus: '2.0'
          memory: 2G
    # ... similar configuration

secrets:
  llm_api_key:
    external: true

networks:
  multiagent-network:
    driver: overlay
    attachable: true
```

## Environment Variables

### Required Variables

```env
# .env file
LLM_BASE_URL=http://host.docker.internal:1234  # For local LM Studio
LLM_API_KEY=
GENERAL_MODEL=llama-3-8b-instruct
MED_MODEL=medgemma-2
```

### Docker-Specific Variables

```env
# Network configuration
DOCKER_NETWORK=multiagent-network
CONTAINER_PREFIX=multiagent

# Resource limits
MEMORY_LIMIT=2g
CPU_LIMIT=2.0

# Logging
LOG_DRIVER=json-file
LOG_MAX_SIZE=10m
LOG_MAX_FILE=3
```

## Building Images

### Build All Images

```bash
# Build all services
docker-compose build

# Build with no cache
docker-compose build --no-cache

# Build specific service
docker-compose build router-agent
```

### Multi-Stage Build

```dockerfile
# Dockerfile with multi-stage build
FROM python:3.11-slim as builder

WORKDIR /app
COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt

FROM python:3.11-slim

WORKDIR /app
COPY --from=builder /root/.local /root/.local
COPY . .

ENV PATH=/root/.local/bin:$PATH
CMD ["python", "launch_a2a_agents.py"]
```

## Container Management

### Starting Services

```bash
# Start all services
docker-compose up -d

# Start specific service
docker-compose up -d router-agent

# View logs
docker-compose logs -f router-agent

# Stop services
docker-compose down
```

### Scaling Services

```bash
# Scale specific service
docker-compose up -d --scale medgemma-agent=3

# Using Docker Swarm
docker service scale multiagent_medgemma-agent=3
```

## Networking

### Internal Communication

Agents communicate using Docker's internal DNS:

```python
# Agents can reach each other by service name
A2A_MEDGEMMA_URL=http://medgemma-agent:9101
A2A_CLINICAL_URL=http://clinical-agent:9102
```

### External Access

For LLM on host machine:

```yaml
# macOS/Windows
LLM_BASE_URL=http://host.docker.internal:1234

# Linux
extra_hosts:
  - "host.docker.internal:host-gateway"
```

## Volume Management

### Persistent Data

```yaml
volumes:
  # For logs
  - ./logs:/app/logs
  
  # For cache
  - cache_volume:/app/.cache
  
  # For configuration
  - ./config:/app/config:ro

volumes:
  cache_volume:
```

## Health Checks

### Container Health

```yaml
healthcheck:
  test: ["CMD", "python", "-c", "import requests; requests.get('http://localhost:9100/health')"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

### Monitoring Stack

```yaml
# docker-compose.monitoring.yml
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
    ports:
      - "9090:9090"
      
  grafana:
    image: grafana/grafana
    ports:
      - "3001:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
```

## Security Considerations

### Non-Root User

```dockerfile
RUN useradd -m -u 1000 appuser
USER appuser
```

### Secrets Management

```bash
# Create Docker secrets
echo "your-api-key" | docker secret create llm_api_key -

# Use in compose
secrets:
  - llm_api_key
```

### Network Isolation

```yaml
networks:
  frontend:
    driver: bridge
  backend:
    driver: bridge
    internal: true
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| **Cannot connect to LLM** | Use `host.docker.internal` or host IP |
| **Port conflicts** | Change ports in docker-compose.yml |
| **Out of memory** | Increase Docker memory limit |
| **Slow builds** | Use `.dockerignore`, multi-stage builds |
| **Container exits** | Check logs: `docker logs <container>` |

### Debug Commands

```bash
# Inspect container
docker inspect multiagent-chat_router-agent_1

# Execute command in container
docker exec -it multiagent-chat_router-agent_1 bash

# View resource usage
docker stats

# Clean up
docker system prune -a
```

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/docker.yml
name: Docker Build

on:
  push:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      
      - name: Build images
        run: docker-compose build
        
      - name: Push to registry
        run: |
          docker tag multiagent-router:latest ${{ secrets.REGISTRY }}/multiagent-router:latest
          docker push ${{ secrets.REGISTRY }}/multiagent-router:latest
```

## Production Deployment

### Docker Swarm

```bash
# Initialize swarm
docker swarm init

# Deploy stack
docker stack deploy -c docker-compose.prod.yml multiagent

# View services
docker service ls

# Update service
docker service update --image multiagent-router:v2 multiagent_router-agent
```

### Kubernetes

```yaml
# k8s/router-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: router-agent
spec:
  replicas: 3
  selector:
    matchLabels:
      app: router-agent
  template:
    metadata:
      labels:
        app: router-agent
    spec:
      containers:
      - name: router
        image: multiagent-router:latest
        ports:
        - containerPort: 9100
        env:
        - name: LLM_BASE_URL
          valueFrom:
            configMapKeyRef:
              name: multiagent-config
              key: llm_base_url
```

## Best Practices

1. **Use specific image tags** instead of `latest`
2. **Implement health checks** for all services
3. **Set resource limits** to prevent runaway containers
4. **Use multi-stage builds** to reduce image size
5. **Externalize configuration** via environment variables
6. **Log to stdout** for container log aggregation
7. **Run as non-root user** for security
8. **Use secrets** for sensitive data

## Next Steps

- [Configure monitoring](monitoring.md) for production
- [Setup Kubernetes](kubernetes.md) for scale
- [Implement CI/CD](../development/cicd.md) pipelines
