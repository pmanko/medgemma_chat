from typing import Dict, Any
import os
import requests
from fastapi import FastAPI

# Native A2A-style router: exposes a single endpoint and a manifest
# For simplicity, we implement a minimal server rather than using SDK scaffolding

ROUTER_PORT = int(os.getenv("A2A_ROUTER_PORT", "9100"))
CLINICAL_URL = os.getenv("A2A_CLINICAL_URL", "http://localhost:9102")
MEDGEMMA_URL = os.getenv("A2A_MEDGEMMA_URL", "http://localhost:9101")

app = FastAPI(title="A2A Router Agent")


@app.get("/.well-known/agent.json")
def agent_card():
    return {
        "agent_id": "user_proxy_agent",
        "name": "User Proxy Router",
        "description": "Routes to medgemma or clinical research skills and returns the final answer.",
        "skills": [
            {
                "name": "route_and_invoke",
                "description": "Choose best skill and return final answer",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "scope": {"type": "string", "enum": ["facility", "hie"]},
                        "facility_id": {"type": "string"},
                        "org_ids": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["query"]
                }
            }
        ]
    }


@app.post("/")
def handle_a2a(message: Dict[str, Any]):
    payload = message.get("payload", {})
    query = payload.get("query", "")
    # naive rule: medical question if mentions symptoms/diagnosis; else clinical_research
    lower = query.lower()
    pick_med = any(k in lower for k in ["symptom", "diagnos", "treatment", "drug", "medication"]) and not any(
        k in lower for k in ["sql", "query", "fhir", "observation", "lab", "vitals", "patient"]
    )

    if pick_med:
        # Call medgemma agent
        resp = requests.post(
            MEDGEMMA_URL.rstrip("/") + "/",
            json={"payload": {"query": query}},
            timeout=90,
        )
        resp.raise_for_status()
        data = resp.json()
    else:
        # Call clinical research agent
        resp = requests.post(
            CLINICAL_URL.rstrip("/") + "/",
            json={"payload": {
                "query": query,
                "scope": payload.get("scope", "hie"),
                "facility_id": payload.get("facility_id"),
                "org_ids": payload.get("org_ids"),
            }},
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()

    # Normalize to A2A-like response
    return {"jsonrpc": "2.0", "result": data.get("result") or data}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=ROUTER_PORT)
