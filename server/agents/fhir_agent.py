import threading
import json
from typing import Optional

import requests
from pyhive import hive

from ..a2a_layer import registry, bus, new_message
from ..config import llm_config, openmrs_config, spark_config
from ..llm_clients import llm_client

AGENT_ID = "clinical_research_agent"
TASK_NAME = "clinical_research"


def run_clinical_research_agent() -> threading.Thread:
    registry.register({
        "agent_id": AGENT_ID,
        "task_name": TASK_NAME,
        "description": "Answers clinical queries using live FHIR and SQL-on-FHIR via Spark Thrift",
        "skills": [
            {
                "name": "clinical_research",
                "description": "Retrieve and synthesize clinical data via FHIR API and SQL-on-FHIR",
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
    })
    bus.register_mailbox(AGENT_ID)

    http = requests.Session()
    spark_conn = {"conn": None}

    def _get_spark_connection():
        if not spark_config.host:
            raise RuntimeError("SPARK_THRIFT_HOST is not configured")
        try:
            if spark_conn["conn"] is None:
                auth = "NONE"
                kwargs = {"host": spark_config.host, "port": spark_config.port, "database": spark_config.database or "default", "auth": auth}
                if spark_config.username and spark_config.password:
                    kwargs.update({"username": spark_config.username, "password": spark_config.password, "auth": "LDAP"})
                spark_conn["conn"] = hive.Connection(**kwargs)
            return spark_conn["conn"]
        except Exception:
            spark_conn["conn"] = None
            raise

    def _ensure_sql_scope(sql: str, scope: str, facility_id: Optional[str], org_ids: Optional[list]) -> bool:
        if scope != "facility":
            return True
        must = [facility_id] if facility_id else []
        must += org_ids or []
        if not must:
            return False
        text = sql.lower()
        return any(mid and mid.lower() in text for mid in must)

    def _ensure_fhir_scope(path: str, scope: str, facility_id: Optional[str], org_ids: Optional[list]) -> bool:
        if scope != "facility":
            return True
        lowered = path.lower()
        tokens = ["organization=", "managingorganization=", "serviceprovider="]
        has_org_param = any(t in lowered for t in tokens)
        has_ident = (facility_id and facility_id.lower() in lowered) or any((oid and oid.lower() in lowered) for oid in (org_ids or []))
        return has_org_param and has_ident

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
        resp = http.get(url, auth=auth, timeout=60)
        resp.raise_for_status()
        return resp.json()

    def _query_spark(sql: str) -> str:
        conn = _get_spark_connection()
        with conn.cursor() as cur:
            cur.execute(sql)
            cols = [desc[0] for desc in cur.description]
            rows = cur.fetchall()
            records = [dict(zip(cols, r)) for r in rows]
            return json.dumps(records[:10])

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
                scope = payload.get("scope", "hie")
                facility_id = payload.get("facility_id")
                org_ids = payload.get("org_ids") or []

                fhir_data = None
                sql_rows = None
                sql_text = None
                fhir_path = None

                if openmrs_config.fhir_base_url:
                    try:
                        fhir_path = _nl_to_fhir_path(user_query, scope, facility_id, org_ids)
                        if not _ensure_fhir_scope(fhir_path, scope, facility_id, org_ids):
                            fhir_path = _nl_to_fhir_path(user_query + " (facility constraints required)", scope, facility_id, org_ids)
                        fhir_data = _query_fhir(fhir_path)
                    except Exception:
                        fhir_data = None

                if spark_config.host:
                    try:
                        sql_text = _nl_to_sql(user_query, scope, facility_id, org_ids)
                        if not _ensure_sql_scope(sql_text, scope, facility_id, org_ids):
                            sql_text = _nl_to_sql(user_query + " (ensure WHERE filters for facility/org)", scope, facility_id, org_ids)
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

                back = new_message(
                    sender_id=AGENT_ID,
                    receiver_id=msg.get("sender_id", "web_ui"),
                    task_name=TASK_NAME,
                    payload={"summary": summary, **combined},
                    correlation_id=msg.get("correlation_id"),
                    status="completed",
                )
                bus.post_message(back)
            except Exception as e:
                back = new_message(
                    sender_id=AGENT_ID,
                    receiver_id=msg.get("sender_id", "web_ui"),
                    task_name=TASK_NAME,
                    payload={"error": str(e)},
                    correlation_id=msg.get("correlation_id"),
                    status="error",
                )
                bus.post_message(back)

    t = threading.Thread(target=loop, name="clinical_research_agent", daemon=True)
    t.start()
    return t 