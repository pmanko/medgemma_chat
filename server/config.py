import os
from dataclasses import dataclass
from typing import Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


@dataclass
class LLMConfig:
    base_url: str = os.getenv("LLM_BASE_URL", "http://localhost:11434/v1")
    api_key: str = os.getenv("LLM_API_KEY", "")
    general_model: str = os.getenv("GENERAL_MODEL", "llama-3-8b-instruct")
    med_model: str = os.getenv("MED_MODEL", "medgemma-2")
    temperature: float = float(os.getenv("LLM_TEMPERATURE", "0.2"))


@dataclass
class OrchestratorConfig:
    provider: str = os.getenv("ORCHESTRATOR_PROVIDER", "openai").lower()
    model: str = os.getenv("ORCHESTRATOR_MODEL", "llama-3-8b-instruct")
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", "")


@dataclass
class AgentConfig:
    enable_a2a: bool = os.getenv("ENABLE_A2A", "true").lower() == "true"
    enable_a2a_native: bool = os.getenv("ENABLE_A2A_NATIVE", "false").lower() == "true"
    chat_timeout_seconds: int = int(os.getenv("CHAT_TIMEOUT_SECONDS", "90"))


@dataclass
class A2AEndpoints:
    router_url: Optional[str] = os.getenv("A2A_ROUTER_URL")
    medgemma_url: Optional[str] = os.getenv("A2A_MEDGEMMA_URL")
    clinical_url: Optional[str] = os.getenv("A2A_CLINICAL_URL")


@dataclass
class OpenMRSConfig:
    fhir_base_url: Optional[str] = os.getenv("OPENMRS_FHIR_BASE_URL")
    auth_username: Optional[str] = os.getenv("OPENMRS_USERNAME")
    auth_password: Optional[str] = os.getenv("OPENMRS_PASSWORD")


@dataclass
class SparkConfig:
    host: Optional[str] = os.getenv("SPARK_THRIFT_HOST")
    port: int = int(os.getenv("SPARK_THRIFT_PORT", "10000"))
    username: Optional[str] = os.getenv("SPARK_THRIFT_USER")
    password: Optional[str] = os.getenv("SPARK_THRIFT_PASSWORD")
    database: Optional[str] = os.getenv("SPARK_THRIFT_DATABASE")


@dataclass
class LocalDatastoreConfig:
    parquet_dir: Optional[str] = os.getenv("FHIR_PARQUET_DIR")


@dataclass
class MySQLConfig:
    """Configuration for OpenMRS MySQL database connection"""
    host: str = os.getenv("MYSQL_HOST", "localhost")
    port: int = int(os.getenv("MYSQL_PORT", "3306"))
    database: str = os.getenv("MYSQL_DATABASE", "openmrs")
    username: str = os.getenv("MYSQL_USER", "openmrs")
    password: str = os.getenv("MYSQL_PASSWORD", "openmrs")


@dataclass
class VectorDBConfig:
    """Configuration for vector database (ChromaDB)"""
    provider: str = os.getenv("VECTOR_DB_PROVIDER", "chromadb")
    persist_directory: str = os.getenv("VECTOR_DB_DIR", "./data/vector_db")
    collection_name: str = os.getenv("VECTOR_COLLECTION", "openmrs_concepts")


@dataclass 
class EmbeddingConfig:
    """Configuration for embedding model and generation"""
    model_name: str = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
    dimension: int = int(os.getenv("EMBEDDING_DIM", "384"))
    batch_size: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))


llm_config = LLMConfig()
orchestrator_config = OrchestratorConfig()
agent_config = AgentConfig()
a2a_endpoints = A2AEndpoints()
openmrs_config = OpenMRSConfig()
spark_config = SparkConfig()
local_config = LocalDatastoreConfig()
mysql_config = MySQLConfig()
vector_db_config = VectorDBConfig()
embedding_config = EmbeddingConfig() 