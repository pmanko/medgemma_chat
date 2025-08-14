import threading
from typing import Dict, Any

from ..a2a_layer import registry, bus, new_message
from ..config import llm_config
from ..llm_clients import llm_client

AGENT_ID = "medgemma_agent"
TASK_NAME = "answer_medical_question"


def run_medgemma_agent() -> threading.Thread:
    registry.register({
        "agent_id": AGENT_ID,
        "task_name": TASK_NAME,
        "description": "Provides expert answers to general medical questions",
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
    })
    bus.register_mailbox(AGENT_ID)

    def loop() -> None:
        while True:
            msg = bus.get_message(AGENT_ID, timeout=None)
            if msg is None:
                continue
            try:
                if msg.get("task_name") != TASK_NAME:
                    continue
                user_query = msg.get("payload", {}).get("query", "")
                messages = [
                    {"role": "system", "content": "You are a medical AI assistant. Provide accurate, evidence-based medical information. Include disclaimers."},
                    {"role": "user", "content": user_query},
                ]
                answer = llm_client.generate_chat(llm_config.med_model, messages, temperature=0.1, max_tokens=800)
                back = new_message(
                    sender_id=AGENT_ID,
                    receiver_id=msg.get("sender_id", "web_ui"),
                    task_name=TASK_NAME,
                    payload={"data": {"answer": answer}},
                    correlation_id=msg.get("correlation_id"),
                    status="completed",
                )
                bus.post_message(back)
            except Exception as e:
                error_back = new_message(
                    sender_id=AGENT_ID,
                    receiver_id=msg.get("sender_id", "web_ui"),
                    task_name=TASK_NAME,
                    payload={"error": str(e)},
                    correlation_id=msg.get("correlation_id"),
                    status="error",
                )
                bus.post_message(error_back)

    t = threading.Thread(target=loop, name="medgemma_agent", daemon=True)
    t.start()
    return t 