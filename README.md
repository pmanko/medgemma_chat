# Multi-Model Chat (A2A-enabled)

Fast, lightweight chat application with two modes:
- Direct per-model chat (general, medical) via external OpenAI-compatible endpoints
- Agents (A2A) mode that semantically routes to specialist agents and optionally queries FHIR sources

References: A2A design and SDK patterns are aligned with the official guides: [Practical Guide to the Official A2A SDK Python](https://a2aprotocol.ai/blog/a2a-sdk-python), [Python A2A: A Comprehensive Guide](https://a2aprotocol.ai/docs/guide/python-a2a.html). For FHIR analytics via Parquet-on-FHIR, see [OHS FHIR Analytics](https://developers.google.com/open-health-stack/fhir-analytics).

## A2A primer (what and why)

Agent2Agent (A2A) is a protocol for interoperable agents. Core ideas:
- Agents are discoverable via a registry/manifest that lists their skills and input schemas.
- Agents communicate via a message bus with a standardized envelope (id, sender, receiver, task/skill, payload, status).
- Orchestrators select a skill and arguments, then invoke the right agent without hard-coding service calls.

This project simulates A2A locally while matching the protocol’s shape (registry + skills + standardized messages) and can run in fully native A2A mode as separate services.

## How we apply A2A here

- Registry and discovery
  - Each agent registers itself (id, description) and advertises its skills with input schemas.
  - `GET /manifest` (simulated mode) exposes current agents and skills.
  - Native mode: each service serves `/.well-known/agent.json`.

- Skills-based orchestration
  - The router (A2A service in native mode) asks the orchestrator LLM to return `{ "skill": string, "args": object }` and invokes the chosen skill.

- Agents and skills (current)
  - `medgemma_agent`: `answer_medical_question(query)`
  - `clinical_research_agent`: `clinical_research(query, scope: facility|hie, facility_id?, org_ids?)`

- Orchestrator options
  - `ORCHESTRATOR_PROVIDER=gemini` or `openai` (OpenAI-compatible local/cloud), configured in the FastAPI bridge.

## Architecture

- Frontend: `client/index.html`, `client/script.js`
- FastAPI bridge: `server/main.py`
  - `/chat` calls router service in native mode or uses in-process simulation when disabled.
- Native A2A services (optional): `server/a2a_services/*`
  - Router: `router_service.py`
  - MedGemma: `medgemma_service.py`
  - Clinical Research: `clinical_service.py`

## Quick Start (native A2A)

Requirements: Python 3.9–3.12

1) Configure environment
```bash
cp env.example .env
# Edit .env with:
# LLM_BASE_URL, LLM_API_KEY, GENERAL_MODEL, MED_MODEL
# OPENMRS_FHIR_BASE_URL, OPENMRS_USERNAME, OPENMRS_PASSWORD
# SPARK_THRIFT_HOST, SPARK_THRIFT_PORT, SPARK_THRIFT_DATABASE, SPARK_THRIFT_USER, SPARK_THRIFT_PASSWORD
# A2A_MEDGEMMA_URL=http://localhost:9101
# A2A_CLINICAL_URL=http://localhost:9102
# A2A_ROUTER_URL=http://localhost:9100
# ENABLE_A2A_NATIVE=true
```

2) Install and run
```bash
poetry install
# Start MedGemma agent
poetry run uvicorn server.a2a_services.medgemma_service:app --host 0.0.0.0 --port 9101
# Start Clinical agent (separate terminal)
poetry run uvicorn server.a2a_services.clinical_service:app --host 0.0.0.0 --port 9102
# Start Router agent (separate terminal)
poetry run uvicorn server.a2a_services.router_service:app --host 0.0.0.0 --port 9100
# Start FastAPI bridge (separate terminal)
poetry run uvicorn server.main:app --host 0.0.0.0 --port 3000
```

3) Open the UI
- Open `client/index.html` in your browser
- Select Agents (A2A) mode; messages go to `/chat`, which forwards to the router agent

## Quick Start (simulated A2A)

```bash
poetry install
poetry run uvicorn server.main:app --host 0.0.0.0 --port 3000
# ENABLE_A2A=true, ENABLE_A2A_NATIVE=false (default) in .env
```

## Environment Configuration

Core LLMs:
- `LLM_BASE_URL` (no `/v1`), `LLM_API_KEY`
- `GENERAL_MODEL`, `MED_MODEL`, `LLM_TEMPERATURE`

Orchestrator (FastAPI bridge):
- `ORCHESTRATOR_PROVIDER` = `gemini` or `openai`
- `ORCHESTRATOR_MODEL`, `GEMINI_API_KEY` (if provider is `gemini`)

A2A mode:
- `ENABLE_A2A=true` enables A2A
- `ENABLE_A2A_NATIVE=true` uses native services; set service URLs:
  - `A2A_ROUTER_URL`, `A2A_MEDGEMMA_URL`, `A2A_CLINICAL_URL`

FHIR + SQL-on-FHIR:
- `OPENMRS_FHIR_BASE_URL`, `OPENMRS_USERNAME`, `OPENMRS_PASSWORD`
- `SPARK_THRIFT_HOST`, `SPARK_THRIFT_PORT`, `SPARK_THRIFT_DATABASE`, `SPARK_THRIFT_USER`, `SPARK_THRIFT_PASSWORD`

## API

- Native manifests: `GET /.well-known/agent.json` (per service)
- Discovery (simulated): `GET /manifest`
- Orchestrated chat: `POST /chat` (FastAPI bridge)
- Direct chat (back-compat): `POST /generate/general`, `POST /generate/medgemma`

## Roadmap

- Phase 2: add a hybrid RAG path with biomedical text embeddings for semantic retrieval over text-heavy FHIR fields; retain MedGemma for clinical synthesis.

## License

MIT