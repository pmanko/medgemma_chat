from typing import Dict, Any, Optional
import os
import json
import requests
from pyhive import hive
from fastapi import FastAPI

from ..config import llm_config, openmrs_config, spark_config
from ..llm_clients import llm_client

PORT = int(os.getenv("A2A_CLINICAL_PORT", "9102"))

app = FastAPI(title="A2A Clinical Research Agent")


@app.get("/.well-known/agent.json")
def agent_card():
    return {
        "agent_id": "clinical_research_agent",
        "name": "Clinical Research Agent",
        "description": "Retrieve and synthesize clinical data via FHIR API and SQL-on-FHIR",
        "skills": [
            {
                "name": "clinical_research",
                "description": "Dual-prompt FHIR + SQL with scope-aware behavior",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "scope": {"type": "string", "enum": ["facility", "hie"], "default": "hie"},
                        "facility_id": {"type": "string"},
                        "org_ids": {"type": "array", "items": {"type": "string"}}
                    },
                    "required": ["query"]
                }
            }
        ]
    }


def _nl_to_fhir_path(user_query: str, scope: str, facility_id: Optional[str], org_ids: Optional[list]) -> str:
    scope_clause = "For facility scope, include organization/managingOrganization/serviceProvider with the given facility/org identifiers." if scope == "facility" else "For HIE scope, do not constrain to a single facility."
    messages = [
        {"role": "system", "content": "Return only a valid FHIR GET path for OpenMRS."},
        {"role": "user", "content": (
            "You generate FHIR search HTTP GET paths for OpenMRS. "
            f"Base URL: {openmrs_config.fhir_base_url or 'UNKNOWN'}\n"
            "Given the user request, produce only the path and query string (no host).\n"
            "Example: /Observation?patient=UUID&_sort=-date&_count=10\n"
            f"Scope: {scope}. {scope_clause}\n"
            f"Facility: {facility_id or ''} | Orgs: {','.join(org_ids or [])}\n"
        ) + f"User request: {user_query}"},
    ]
    path = llm_client.generate_chat(llm_config.general_model, messages, temperature=0.0, max_tokens=160).strip().splitlines()[0]
    if not path.startswith("/"):
        path = "/" + path
    return path


def _nl_to_sql(user_query: str, scope: str, facility_id: Optional[str], org_ids: Optional[list]) -> str:
    scope_clause = (
        "When scope is facility, include WHERE facility_id = '{fid}' or org_id IN ({orgs}) as appropriate."
    ).format(fid=facility_id or "", orgs=",".join([f"'{o}'" for o in (org_ids or [])])) if scope == "facility" else "Do not constrain to a single facility (HIE scope)."
    messages = [
        {"role": "system", "content": "Return only one safe SQL SELECT statement for Spark SQL (limit 50)."},
        {"role": "user", "content": (
            "Write SQL to query SQL-on-FHIR views (observation, patient, encounter). Limit to 50 rows. "
            f"Scope: {scope}. {scope_clause}\n"
            f"Facility: {facility_id or ''} | Orgs: {','.join(org_ids or [])}\n"
            f"Request: {user_query}"
        )},
    ]
    sql = llm_client.generate_chat(llm_config.general_model, messages, temperature=0.0, max_tokens=256)
    sql = sql.strip().split(";")[0]
    if " limit " not in sql.lower():
        sql += " LIMIT 50"
    return sql


def _query_fhir(path: str):
    if not openmrs_config.fhir_base_url:
        raise RuntimeError("OPENMRS_FHIR_BASE_URL is not configured")
    url = openmrs_config.fhir_base_url.rstrip("/") + path
    auth = (openmrs_config.auth_username, openmrs_config.auth_password) if (openmrs_config.auth_username and openmrs_config.auth_password) else None
    resp = requests.get(url, auth=auth, timeout=60)
    resp.raise_for_status()
    return resp.json()


def _query_spark(sql: str) -> str:
    conn = hive.Connection(
        host=spark_config.host,
        port=spark_config.port,
        database=spark_config.database or "default",
        auth=("LDAP" if (spark_config.username and spark_config.password) else "NONE"),
        username=(spark_config.username or None),
        password=(spark_config.password or None),
    )
    with conn.cursor() as cur:
        cur.execute(sql)
        cols = [desc[0] for desc in cur.description]
        rows = cur.fetchall()
        records = [dict(zip(cols, r)) for r in rows]
        return json.dumps(records[:10])


@app.post("/")
def handle_a2a(message: Dict[str, Any]):
    payload = message.get("payload", {})
    query = payload.get("query", "")
    scope = payload.get("scope", "hie")
    facility_id = payload.get("facility_id")
    org_ids = payload.get("org_ids") or []

    fhir_data = None
    sql_rows = None
    sql_text = None
    fhir_path = None

    try:
        fhir_path = _nl_to_fhir_path(query, scope, facility_id, org_ids)
        fhir_data = _query_fhir(fhir_path)
    except Exception:
        fhir_data = None

    try:
        if not spark_config.host:
            raise RuntimeError("SPARK_THRIFT_HOST is not configured")
        sql_text = _nl_to_sql(query, scope, facility_id, org_ids)
        sql_rows = _query_spark(sql_text)
    except Exception:
        sql_rows = None

    combined = {"fhir": fhir_data, "sql_rows": sql_rows, "fhir_path": fhir_path, "sql": sql_text, "scope": scope}
    summary = llm_client.generate_chat(
        llm_config.general_model,
        [
            {"role": "system", "content": "Combine and summarize clinical evidence succinctly for a clinician."},
            {"role": "user", "content": json.dumps(combined)[:6000]},
        ],
        temperature=0.2,
        max_tokens=320,
    )

    return {"jsonrpc": "2.0", "result": {"content": {"text": summary}}}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
