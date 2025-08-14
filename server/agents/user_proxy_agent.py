import threading
import json

from ..a2a_layer import registry, bus, new_message
from ..llm_clients import orchestrator_client

AGENT_ID = "user_proxy_agent"
TASK_NAME = "initiate_query"


def _routing_prompt(user_query: str, skills_description: str) -> str:
    return (
        "You are an orchestrator. Select the best skill and arguments for this query based on the available skills.\n"
        "Return strict JSON: {\"skill\": string, \"args\": object}. Do not include explanations.\n"
        f"Available skills:\n{skills_description}\n\n"
        f"User query: {user_query}\n"
    )


def run_user_proxy_agent():
    registry.register({
        "agent_id": AGENT_ID,
        "task_name": TASK_NAME,
        "description": "Routes user queries to the right specialist skill and selects args",
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
                payload = msg.get("payload", {})
                user_query = payload.get("query", "")

                # Build a skills listing from registry entries
                agents = registry.list_agents()
                skills_lines = []
                for a in agents:
                    for sk in a.get("skills", []):
                        skills_lines.append(f"- {a['agent_id']}.{sk.get('name')}: {sk.get('description','')}; input_schema keys: {','.join(sk.get('input_schema',{}).get('properties',{}).keys())}")
                skills_desc = "\n".join(skills_lines) if skills_lines else "- medgemma_agent.answer_medical_question; - clinical_research_agent.clinical_research"

                messages = [
                    {"role": "system", "content": "Return only valid JSON. Do not include explanations."},
                    {"role": "user", "content": _routing_prompt(user_query, skills_desc)},
                ]
                decision_text = orchestrator_client.route(messages, temperature=0.0, max_tokens=128)
                try:
                    decision = json.loads(decision_text)
                except Exception:
                    # Fallback minimal structure
                    decision = {"skill": (decision_text or "").strip().split()[0], "args": {}}

                skill = str(decision.get("skill", "")).strip()
                args = decision.get("args") or {}

                # Map skill to task_name (1:1 by convention)
                task_name = skill
                candidates = registry.lookup(task_name)
                if not candidates:
                    raise RuntimeError(f"No agent available for skill '{task_name}'")
                target = candidates[0]

                forward_payload = dict(payload)
                # Merge orchestrator-selected args (e.g., scope/facility/org_ids)
                forward_payload.update(args)

                forward = new_message(
                    sender_id=AGENT_ID,
                    receiver_id=target["agent_id"],
                    task_name=task_name,
                    payload=forward_payload,
                    correlation_id=msg.get("correlation_id"),
                )
                bus.post_message(forward)
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

    t = threading.Thread(target=loop, name="user_proxy_agent", daemon=True)
    t.start()
    return t 