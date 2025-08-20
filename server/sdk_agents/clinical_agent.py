"""
Clinical Research Agent using A2A SDK
Queries and synthesizes data from FHIR APIs and SQL-on-FHIR databases
"""

from a2a import Agent
from a2a.types import AgentCard, Skill, InputSchema, OutputSchema
from a2a.decorators import skill
import httpx
import logging
from typing import Dict, Any, Optional, List
import os
import json

logger = logging.getLogger(__name__)

class ClinicalResearchAgent(Agent):
    """Agent for querying and analyzing clinical data from FHIR and SQL sources"""
    
    def __init__(self):
        super().__init__()
        
        # LLM configuration
        self.llm_base_url = os.getenv("LLM_BASE_URL", "http://localhost:1234")
        self.llm_api_key = os.getenv("LLM_API_KEY", "")
        self.general_model = os.getenv("GENERAL_MODEL", "llama-3-8b-instruct")
        self.med_model = os.getenv("MED_MODEL", "medgemma-2")
        
        # FHIR configuration
        self.fhir_base_url = os.getenv("OPENMRS_FHIR_BASE_URL")
        self.fhir_username = os.getenv("OPENMRS_USERNAME")
        self.fhir_password = os.getenv("OPENMRS_PASSWORD")
        
        # Spark SQL configuration
        self.spark_host = os.getenv("SPARK_THRIFT_HOST")
        self.spark_port = int(os.getenv("SPARK_THRIFT_PORT", "10000"))
        self.spark_database = os.getenv("SPARK_THRIFT_DATABASE", "default")
        
        self.http_client = httpx.AsyncClient(timeout=120.0)
        logger.info(f"Clinical Research agent initialized - FHIR: {bool(self.fhir_base_url)}, Spark: {bool(self.spark_host)}")
    
    async def get_agent_card(self) -> AgentCard:
        """Return agent capabilities following A2A protocol"""
        return AgentCard(
            name="Clinical Research Assistant",
            description="Queries and synthesizes clinical data from FHIR APIs and SQL-on-FHIR databases",
            version="1.0.0",
            skills=[
                Skill(
                    name="clinical_research",
                    description="Retrieve and analyze clinical data with scope-based access control",
                    input_schema=InputSchema(
                        type="object",
                        properties={
                            "query": {
                                "type": "string",
                                "description": "Clinical research question or data request"
                            },
                            "scope": {
                                "type": "string",
                                "enum": ["facility", "hie"],
                                "default": "hie",
                                "description": "Access scope: facility-specific or health information exchange"
                            },
                            "facility_id": {
                                "type": "string",
                                "description": "Facility identifier for facility-scoped queries"
                            },
                            "org_ids": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Organization identifiers for filtering"
                            },
                            "data_source": {
                                "type": "string",
                                "enum": ["auto", "fhir", "sql"],
                                "default": "auto",
                                "description": "Preferred data source"
                            }
                        },
                        required=["query"]
                    ),
                    output_schema=OutputSchema(
                        type="object",
                        properties={
                            "response": {"type": "string", "description": "Synthesized clinical response"},
                            "data_source": {"type": "string", "description": "Data source used"},
                            "scope": {"type": "string", "description": "Access scope applied"},
                            "records_found": {"type": "integer", "description": "Number of records found"},
                            "query_executed": {"type": "string", "description": "The actual query executed"},
                            "raw_data": {"type": "object", "description": "Raw data retrieved (optional)"}
                        },
                        required=["response", "data_source"]
                    )
                )
            ]
        )
    
    @skill("clinical_research")
    async def clinical_research(
        self,
        query: str,
        scope: str = "hie",
        facility_id: Optional[str] = None,
        org_ids: Optional[List[str]] = None,
        data_source: str = "auto"
    ) -> Dict[str, Any]:
        """
        Execute clinical research query with scope-based access control
        
        Args:
            query: Clinical research question
            scope: Access scope (facility or hie)
            facility_id: Facility identifier for facility scope
            org_ids: Organization identifiers
            data_source: Preferred data source (auto, fhir, sql)
            
        Returns:
            Dictionary with synthesized response and metadata
        """
        logger.info(f"Processing clinical research query: {query[:100]}... (scope: {scope}, source: {data_source})")
        
        # Validate scope requirements
        if scope == "facility" and not facility_id and not org_ids:
            raise ValueError("Facility scope requires facility_id or org_ids")
        
        # Determine data source
        if data_source == "auto":
            data_source = await self._select_data_source(query)
        
        # Execute query based on data source
        if data_source == "fhir" and self.fhir_base_url:
            query_result = await self._query_fhir(query, scope, facility_id, org_ids)
        elif data_source == "sql" and self.spark_host:
            query_result = await self._query_sql(query, scope, facility_id, org_ids)
        else:
            # Fallback to mock data if no real data source configured
            query_result = await self._query_mock(query, scope, facility_id, org_ids)
        
        # Synthesize response using MedGemma
        synthesis = await self._synthesize_response(
            query, 
            query_result["data"],
            query_result["query_executed"]
        )
        
        return {
            "response": synthesis,
            "data_source": query_result["source"],
            "scope": scope,
            "records_found": query_result["record_count"],
            "query_executed": query_result["query_executed"],
            "raw_data": query_result["data"] if len(str(query_result["data"])) < 1000 else None
        }
    
    async def _select_data_source(self, query: str) -> str:
        """Use LLM to determine best data source for query"""
        fhir_keywords = ["patient", "observation", "medication", "encounter", "condition", "procedure"]
        sql_keywords = ["aggregate", "count", "average", "trend", "statistics", "report"]
        
        query_lower = query.lower()
        fhir_score = sum(1 for kw in fhir_keywords if kw in query_lower)
        sql_score = sum(1 for kw in sql_keywords if kw in query_lower)
        
        if fhir_score > sql_score:
            return "fhir"
        elif sql_score > 0:
            return "sql"
        else:
            return "fhir"  # Default to FHIR
    
    async def _query_fhir(
        self, 
        query: str, 
        scope: str, 
        facility_id: Optional[str], 
        org_ids: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Generate and execute FHIR API query"""
        
        # Generate FHIR query using LLM
        fhir_path = await self._generate_fhir_query(query, scope, facility_id, org_ids)
        
        # Validate scope constraints
        if scope == "facility" and not self._validate_fhir_scope(fhir_path, facility_id, org_ids):
            raise ValueError("Generated FHIR query does not meet facility scope requirements")
        
        # Execute FHIR query
        headers = {}
        if self.fhir_username and self.fhir_password:
            import base64
            auth = base64.b64encode(f"{self.fhir_username}:{self.fhir_password}".encode()).decode()
            headers["Authorization"] = f"Basic {auth}"
        
        try:
            response = await self.http_client.get(
                f"{self.fhir_base_url}{fhir_path}",
                headers=headers
            )
            response.raise_for_status()
            data = response.json()
            
            # Extract relevant data from FHIR bundle
            resources = []
            if "entry" in data:
                resources = [entry.get("resource", {}) for entry in data.get("entry", [])]
            
            return {
                "source": "FHIR",
                "query_executed": fhir_path,
                "data": resources,
                "record_count": len(resources)
            }
            
        except Exception as e:
            logger.error(f"FHIR query failed: {e}")
            # Return mock data as fallback
            return await self._query_mock(query, scope, facility_id, org_ids)
    
    async def _query_sql(
        self, 
        query: str, 
        scope: str, 
        facility_id: Optional[str], 
        org_ids: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Generate and execute SQL-on-FHIR query"""
        
        # Generate SQL query using LLM
        sql_query = await self._generate_sql_query(query, scope, facility_id, org_ids)
        
        # Validate scope constraints
        if scope == "facility" and not self._validate_sql_scope(sql_query, facility_id, org_ids):
            raise ValueError("Generated SQL query does not meet facility scope requirements")
        
        try:
            # Execute via PyHive if available
            from pyhive import hive
            
            conn = hive.Connection(
                host=self.spark_host,
                port=self.spark_port,
                database=self.spark_database
            )
            
            cursor = conn.cursor()
            cursor.execute(sql_query)
            columns = [desc[0] for desc in cursor.description]
            rows = cursor.fetchall()
            
            # Convert to list of dicts
            data = [dict(zip(columns, row)) for row in rows]
            
            cursor.close()
            conn.close()
            
            return {
                "source": "SQL-on-FHIR",
                "query_executed": sql_query,
                "data": data,
                "record_count": len(data)
            }
            
        except ImportError:
            logger.warning("PyHive not available, using mock data")
            return await self._query_mock(query, scope, facility_id, org_ids)
        except Exception as e:
            logger.error(f"SQL query failed: {e}")
            return await self._query_mock(query, scope, facility_id, org_ids)
    
    async def _query_mock(
        self, 
        query: str, 
        scope: str, 
        facility_id: Optional[str], 
        org_ids: Optional[List[str]]
    ) -> Dict[str, Any]:
        """Return mock data for testing"""
        
        mock_data = {
            "patients": [
                {"id": "P001", "name": "John Doe", "age": 45, "facility": facility_id or "F001"},
                {"id": "P002", "name": "Jane Smith", "age": 32, "facility": facility_id or "F001"}
            ],
            "observations": [
                {"patient": "P001", "type": "blood_pressure", "value": "120/80", "date": "2024-01-15"},
                {"patient": "P001", "type": "heart_rate", "value": "72", "date": "2024-01-15"}
            ]
        }
        
        return {
            "source": "Mock Data",
            "query_executed": f"MOCK: {query[:50]}...",
            "data": mock_data,
            "record_count": 2
        }
    
    async def _generate_fhir_query(
        self, 
        query: str, 
        scope: str, 
        facility_id: Optional[str], 
        org_ids: Optional[List[str]]
    ) -> str:
        """Use LLM to generate FHIR API query path"""
        
        scope_instruction = ""
        if scope == "facility":
            scope_instruction = f"Include organization filter for facility {facility_id} or orgs {org_ids}"
        
        prompt = f"""Generate a FHIR R4 API query path for this request:
        Request: {query}
        Scope: {scope} {scope_instruction}
        
        Return only the path and query parameters, starting with /
        Examples: /Patient?name=John, /Observation?patient=123&code=loinc|1234
        
        Path:"""
        
        headers = {"Content-Type": "application/json"}
        if self.llm_api_key:
            headers["Authorization"] = f"Bearer {self.llm_api_key}"
        
        response = await self.http_client.post(
            f"{self.llm_base_url}/v1/chat/completions",
            json={
                "model": self.general_model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 200
            },
            headers=headers
        )
        
        data = response.json()
        path = data["choices"][0]["message"]["content"].strip()
        
        # Ensure path starts with /
        if not path.startswith("/"):
            path = "/" + path
            
        return path
    
    async def _generate_sql_query(
        self, 
        query: str, 
        scope: str, 
        facility_id: Optional[str], 
        org_ids: Optional[List[str]]
    ) -> str:
        """Use LLM to generate SQL-on-FHIR query"""
        
        scope_clause = ""
        if scope == "facility":
            if facility_id:
                scope_clause = f"WHERE facility_id = '{facility_id}'"
            elif org_ids:
                scope_clause = f"WHERE org_id IN ({','.join([f\"'{o}'\" for o in org_ids])})"
        
        prompt = f"""Generate a SQL query for Parquet-on-FHIR tables:
        Request: {query}
        Available tables: patient, observation, condition, medication_request
        {f'Required filter: {scope_clause}' if scope_clause else ''}
        
        Return only the SQL query.
        
        Query:"""
        
        headers = {"Content-Type": "application/json"}
        if self.llm_api_key:
            headers["Authorization"] = f"Bearer {self.llm_api_key}"
        
        response = await self.http_client.post(
            f"{self.llm_base_url}/v1/chat/completions",
            json={
                "model": self.general_model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.1,
                "max_tokens": 300
            },
            headers=headers
        )
        
        data = response.json()
        sql = data["choices"][0]["message"]["content"].strip()
        
        # Remove markdown formatting if present
        if sql.startswith("```"):
            sql = sql.split("```")[1].replace("sql", "").strip()
            
        return sql
    
    async def _synthesize_response(self, query: str, data: Any, query_executed: str) -> str:
        """Use MedGemma to synthesize clinical response from data"""
        
        # Prepare data summary for synthesis
        data_summary = json.dumps(data, indent=2)[:2000]  # Limit size
        
        prompt = f"""Based on the following clinical data, provide a comprehensive answer to this question:
        
        Question: {query}
        
        Query executed: {query_executed}
        
        Data retrieved:
        {data_summary}
        
        Provide a clear, clinically-relevant summary of the findings."""
        
        headers = {"Content-Type": "application/json"}
        if self.llm_api_key:
            headers["Authorization"] = f"Bearer {self.llm_api_key}"
        
        response = await self.http_client.post(
            f"{self.llm_base_url}/v1/chat/completions",
            json={
                "model": self.med_model,
                "messages": [
                    {"role": "system", "content": "You are a clinical data analyst. Synthesize the data into clear, actionable insights."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.2,
                "max_tokens": 800
            },
            headers=headers
        )
        
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    
    def _validate_fhir_scope(self, path: str, facility_id: Optional[str], org_ids: Optional[List[str]]) -> bool:
        """Validate FHIR query meets scope requirements"""
        if not facility_id and not org_ids:
            return True  # No facility scope to validate
            
        path_lower = path.lower()
        has_org_param = any(param in path_lower for param in ["organization=", "managingorganization=", "serviceprovider="])
        
        if not has_org_param:
            return False
            
        # Check if facility/org ID is in the query
        ids_to_check = [facility_id] if facility_id else []
        ids_to_check.extend(org_ids or [])
        
        return any(id_val and id_val.lower() in path_lower for id_val in ids_to_check)
    
    def _validate_sql_scope(self, sql: str, facility_id: Optional[str], org_ids: Optional[List[str]]) -> bool:
        """Validate SQL query meets scope requirements"""
        if not facility_id and not org_ids:
            return True  # No facility scope to validate
            
        sql_lower = sql.lower()
        
        # Check for WHERE clause with facility/org filter
        if "where" not in sql_lower:
            return False
            
        ids_to_check = [facility_id] if facility_id else []
        ids_to_check.extend(org_ids or [])
        
        return any(id_val and id_val.lower() in sql_lower for id_val in ids_to_check)
    
    async def cleanup(self):
        """Cleanup resources on shutdown"""
        await self.http_client.aclose()
        logger.info("Clinical Research agent cleanup completed")


# For standalone testing
if __name__ == "__main__":
    import asyncio
    from a2a.server import AgentServer
    import uvicorn
    
    async def main():
        agent = ClinicalResearchAgent()
        server = AgentServer(agent)
        
        # Test the agent
        card = await agent.get_agent_card()
        print(f"Agent: {card.name}")
        print(f"Skills: {[s.name for s in card.skills]}")
        
        # Run server
        config = uvicorn.Config(
            server.app,
            host="0.0.0.0",
            port=9102,
            log_level="info"
        )
        server = uvicorn.Server(config)
        await server.serve()
    
    asyncio.run(main())
