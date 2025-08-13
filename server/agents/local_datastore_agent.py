import threading
import os
from typing import Dict, Any, Optional

import duckdb

from ..a2a_layer import registry, bus, new_message
from ..config import llm_config, local_config
from ..llm_clients import llm_client

AGENT_ID = "local_datastore_agent"
TASK_NAME = "query_local_fhir_datastore"


def _sql_prompt(user_query: str, parquet_dir: Optional[str]) -> str:
    return (
        "You write safe SQL for DuckDB to query FHIR Parquet files in a directory.\n"
        "Assume tables are auto-detected from Parquet files, and common FHIR resources exist as tables like observation, patient, encounter.\n"
        "Return only one SQL SELECT statement limited to 50 rows.\n"
        f"Parquet dir: {parquet_dir or 'UNKNOWN'}\n"
        f"Request: {user_query}"
    )


def run_local_datastore_agent() -> threading.Thread:
    registry.register({
        "agent_id": AGENT_ID,
        "task_name": TASK_NAME,
        "description": "Queries local FHIR Parquet with DuckDB and summarizes the result",
    })
    bus.register_mailbox(AGENT_ID)

    con = duckdb.connect()

    def loop() -> None:
        while True:
            msg = bus.get_message(AGENT_ID, timeout=None)
            if msg is None:
                continue
            try:
                if msg.get("task_name") != TASK_NAME:
                    continue
                user_query = msg.get("payload", {}).get("query", "")
                if not local_config.parquet_dir or not os.path.isdir(local_config.parquet_dir):
                    raise RuntimeError("FHIR_PARQUET_DIR is not configured or does not exist")

                # 1) NL -> SQL
                sql = llm_client.generate_chat(
                    llm_config.general_model,
                    [
                        {"role": "system", "content": "Return only one safe SQL SELECT statement for DuckDB."},
                        {"role": "user", "content": _sql_prompt(user_query, local_config.parquet_dir)},
                    ],
                    temperature=0.0,
                    max_tokens=256,
                )
                sql = sql.strip().split(";")[0] + " LIMIT 50"

                # 2) Query DuckDB with Parquet files as tables via glob
                con.execute("CREATE OR REPLACE VIEW observation AS SELECT * FROM read_parquet(?);", [os.path.join(local_config.parquet_dir, "Observation*.parquet")])
                con.execute("CREATE OR REPLACE VIEW patient AS SELECT * FROM read_parquet(?);", [os.path.join(local_config.parquet_dir, "Patient*.parquet")])
                con.execute("CREATE OR REPLACE VIEW encounter AS SELECT * FROM read_parquet(?);", [os.path.join(local_config.parquet_dir, "Encounter*.parquet")])

                rows = con.execute(sql).fetch_df(limit=50)
                preview = rows.head(10).to_json(orient="records")

                summary = llm_client.generate_chat(
                    llm_config.general_model,
                    [
                        {"role": "system", "content": "Summarize tabular results for a clinician in 2-3 sentences."},
                        {"role": "user", "content": preview},
                    ],
                    temperature=0.2,
                    max_tokens=200,
                )

                back = new_message(
                    sender_id=AGENT_ID,
                    receiver_id=msg.get("sender_id", "web_ui"),
                    task_name=TASK_NAME,
                    payload={"sql": sql, "rows": preview, "summary": summary},
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

    t = threading.Thread(target=loop, name="local_datastore_agent", daemon=True)
    t.start()
    return t 