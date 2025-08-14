from typing import Dict, Any
import os
import requests
from fastapi import FastAPI

from ..config import llm_config

PORT = int(os.getenv("A2A_MEDGEMMA_PORT", "9101"))

app = FastAPI(title="A2A MedGemma Agent")


@app.get("/.well-known/agent.json")
def agent_card():
    return {
        "agent_id": "medgemma_agent",
        "name": "MedGemma Agent",
        "description": "Medical Q&A synthesis",
        "skills": [
            {
                "name": "answer_medical_question",
                "description": "Synthesize clinically sound answers to medical questions",
                "input_schema": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"]
                }
            }
        ]
    }


@app.post("/")
def handle_a2a(message: Dict[str, Any]):
    query = (message.get("payload") or {}).get("query", "")
    # Call external LLM (MedGemma model name)
    resp = requests.post(
        llm_config.base_url.rstrip("/") + "/v1/chat/completions",
        headers={"Content-Type": "application/json", **({"Authorization": f"Bearer {llm_config.api_key}"} if llm_config.api_key else {})},
        json={
            "model": llm_config.med_model,
            "messages": [
                {"role": "system", "content": "You are a medical AI assistant. Provide accurate, evidence-based information. Include a brief disclaimer."},
                {"role": "user", "content": query}
            ],
            "temperature": 0.1,
            "stream": False,
            "max_tokens": 800,
        },
        timeout=90,
    )
    resp.raise_for_status()
    data = resp.json()
    text = data["choices"][0]["message"]["content"].strip()
    return {"jsonrpc": "2.0", "result": {"content": {"text": text}}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
