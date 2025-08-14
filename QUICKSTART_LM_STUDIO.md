## Quick Start: Run A2A Services with LM Studio

**Goal:** Run the native A2A agents locally while using LM Studio’s OpenAI‑compatible API for all LLM calls.

### 1) LM Studio version and mode

**Use LM Studio v0.3.5 or newer.** This adds on‑demand model loading so the `model` field is honored and models can be pulled/loaded automatically. The latest releases (e.g., 0.3.20) are recommended for stability and performance.

**Run the Local API server.** Either enable the server in the app UI or run LM Studio in headless mode per docs. The API is OpenAI‑compatible and served on a port you choose (commonly 1234).

- Headless/API reference: `https://lmstudio.ai/docs/app/api/headless`
- On‑demand model loading: `https://lmstudio.ai/blog/lmstudio-v0.3.5#on-demand-model-loading`

### 2) Configure environment

**Create `.env` from `env.example`.** Point the base URL to your LM Studio server (without `/v1`). Set distinct models if you want the router, medical, and general roles to use different models.

```env
LLM_BASE_URL=http://localhost:1234
LLM_API_KEY=
GENERAL_MODEL=llama-3-8b-instruct
MED_MODEL=llama-3-8b-instruct
LLM_TEMPERATURE=0.2

# Orchestrator can also use LM Studio
ORCHESTRATOR_PROVIDER=openai
ORCHESTRATOR_MODEL=llama-3-8b-instruct

ENABLE_A2A=true
ENABLE_A2A_NATIVE=true
A2A_MEDGEMMA_URL=http://localhost:9101
A2A_CLINICAL_URL=http://localhost:9102
A2A_ROUTER_URL=http://localhost:9100

# Optional data sources
OPENMRS_FHIR_BASE_URL=http://your-openmrs/fhir
SPARK_THRIFT_HOST=localhost
SPARK_THRIFT_PORT=10000
SPARK_THRIFT_DATABASE=default
```

### 3) Start services

**Use separate terminals.** Install, then run each A2A service and the bridge.

```bash
poetry install

# MedGemma agent (A2A)
poetry run uvicorn server.a2a_services.medgemma_service:app --host 0.0.0.0 --port 9101

# Clinical research agent (A2A)
poetry run uvicorn server.a2a_services.clinical_service:app --host 0.0.0.0 --port 9102

# Router agent (A2A)
poetry run uvicorn server.a2a_services.router_service:app --host 0.0.0.0 --port 9100

# FastAPI bridge (UI → /chat)
poetry run uvicorn server.main:app --host 0.0.0.0 --port 3000
```

### 4) Use the UI

**Open `client/index.html`.** Switch to **Agents (A2A)** mode. Messages go to `/chat`, which forwards to the router service.

### 5) Verify

**Manifests:** `http://localhost:9100/.well-known/agent.json` (router), `:9101/.well-known/agent.json` (medgemma), `:9102/.well-known/agent.json` (clinical).  
**Bridge health:** `http://localhost:3000/health`.

### Notes and best practices

**On‑demand loading (v0.3.5+).** With on‑demand loading enabled, LM Studio honors the `model` field. You can set different `GENERAL_MODEL`, `MED_MODEL`, and `ORCHESTRATOR_MODEL`. The first request may take longer while the model is pulled and loaded. See: `https://lmstudio.ai/blog/lmstudio-v0.3.5#on-demand-model-loading`.

**Headless mode.** If you prefer running without the UI, follow: `https://lmstudio.ai/docs/app/api/headless`. Ensure the server binds to the expected host/port and (if required) enable remote access and/or API key.

**Performance (GPU).** For NVIDIA RTX GPUs, install a recent CUDA (e.g., 12.8) and enable GPU offloading/optimizations in LM Studio. Choose an appropriate quantization and consider enabling flash‑attention if supported.

**Model selection.** If you keep a single model loaded, LM Studio may ignore `model` and use the active one. With on‑demand loading, distinct model names per role are supported. For strict separation, run multiple LM Studio instances on different ports or use an API server that fully enforces `model`.

**API compatibility.** Our backend calls the OpenAI `/v1/chat/completions` endpoint. Keep `LLM_BASE_URL` without `/v1`; the code appends it.

**Troubleshooting.** If responses are empty, confirm the LM Studio server is running, reachable, and a model is available. Timeouts on clinical research usually indicate FHIR or Spark misconfiguration; verify those endpoints.
