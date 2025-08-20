"""
Configuration management for the Medical Multi-Agent Chat System.
Provides smart defaults for all settings to minimize configuration burden.
"""

import os
    from dotenv import load_dotenv

# Load environment variables from .env file
    load_dotenv()

# ==============================================================================
# LLM Configuration
# ==============================================================================

# LM Studio endpoint (required)
LLM_BASE_URL = os.getenv("LLM_BASE_URL", "http://localhost:1234")
LLM_API_KEY = os.getenv("LLM_API_KEY", "")  # Empty for local LM Studio

# Model selection
GENERAL_MODEL = os.getenv("GENERAL_MODEL", "llama-3-8b-instruct")
MED_MODEL = os.getenv("MED_MODEL", GENERAL_MODEL)  # Defaults to same as general

# LLM parameters (smart defaults, not exposed in env.example)
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "1000"))

# ==============================================================================
# Orchestrator Configuration (Optional Gemini)
# ==============================================================================

# Provider can be "openai" (for LM Studio) or "gemini"
ORCHESTRATOR_PROVIDER = os.getenv("ORCHESTRATOR_PROVIDER", "openai")

if ORCHESTRATOR_PROVIDER == "gemini":
    # Use Google Gemini for orchestration
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY required when ORCHESTRATOR_PROVIDER=gemini")
    ORCHESTRATOR_MODEL = os.getenv("ORCHESTRATOR_MODEL", "gemini-1.5-flash")
else:
    # Use local LM Studio for orchestration
    ORCHESTRATOR_MODEL = GENERAL_MODEL
    GEMINI_API_KEY = None

# ==============================================================================
# A2A Service Configuration
# ==============================================================================

# Agent service URLs (defaults work for standard setup)
A2A_ROUTER_URL = os.getenv("A2A_ROUTER_URL", "http://localhost:9100")
A2A_MEDGEMMA_URL = os.getenv("A2A_MEDGEMMA_URL", "http://localhost:9101")
A2A_CLINICAL_URL = os.getenv("A2A_CLINICAL_URL", "http://localhost:9102")

# A2A is always enabled in SDK version
ENABLE_A2A = True
ENABLE_A2A_NATIVE = True

# ==============================================================================
# Clinical Data Sources (Optional)
# ==============================================================================

# OpenMRS FHIR configuration
OPENMRS_FHIR_BASE_URL = os.getenv("OPENMRS_FHIR_BASE_URL", "")
OPENMRS_USERNAME = os.getenv("OPENMRS_USERNAME", "admin")
OPENMRS_PASSWORD = os.getenv("OPENMRS_PASSWORD", "Admin123")

# Local FHIR Parquet files
FHIR_PARQUET_DIR = os.getenv("FHIR_PARQUET_DIR", "")

# Spark SQL configuration (advanced users only)
SPARK_THRIFT_HOST = os.getenv("SPARK_THRIFT_HOST", "")
SPARK_THRIFT_PORT = int(os.getenv("SPARK_THRIFT_PORT", "10000"))
SPARK_THRIFT_DATABASE = os.getenv("SPARK_THRIFT_DATABASE", "default")

# ==============================================================================
# Application Settings (Smart Defaults)
# ==============================================================================

# Timeouts
CHAT_TIMEOUT_SECONDS = int(os.getenv("CHAT_TIMEOUT_SECONDS", "90"))
LLM_TIMEOUT_SECONDS = int(os.getenv("LLM_TIMEOUT_SECONDS", "60"))
AGENT_STARTUP_TIMEOUT = int(os.getenv("AGENT_STARTUP_TIMEOUT", "10"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# Performance
MAX_CONCURRENT_REQUESTS = int(os.getenv("MAX_CONCURRENT_REQUESTS", "10"))
CONNECTION_POOL_SIZE = int(os.getenv("CONNECTION_POOL_SIZE", "20"))

# ==============================================================================
# Security Settings (Defaults for Local Development)
# ==============================================================================

# CORS - Allow common local development origins
CORS_ORIGINS = [
    "http://localhost:3000",  # React dev server
    "http://localhost:8080",  # Production UI
    "http://localhost:7860",  # Alternative port
]

# SSL/TLS - Disabled for local development
USE_HTTPS = os.getenv("USE_HTTPS", "false").lower() == "true"

# API Authentication - Disabled by default for ease of use
API_KEY_REQUIRED = os.getenv("API_KEY_REQUIRED", "false").lower() == "true"
API_KEY = os.getenv("API_KEY", "")

# ==============================================================================
# Development/Debug Settings
# ==============================================================================

# Environment mode
ENV = os.getenv("ENV", "development")
DEBUG = ENV == "development"

# Auto-reload for development
RELOAD = os.getenv("RELOAD", str(DEBUG)).lower() == "true"

# ==============================================================================
# Validation
# ==============================================================================

def validate_config():
    """Validate configuration and provide helpful error messages."""
    errors = []
    
    # Check required settings
    if not LLM_BASE_URL:
        errors.append("LLM_BASE_URL is required. Set it to your LM Studio endpoint (e.g., http://localhost:1234)")
    
    if not GENERAL_MODEL:
        errors.append("GENERAL_MODEL is required. Set it to your model name in LM Studio")
    
    # Check optional features
    if OPENMRS_FHIR_BASE_URL and not OPENMRS_USERNAME:
        errors.append("OPENMRS_USERNAME required when using OpenMRS FHIR")
    
    if errors:
        print("Configuration errors:")
        for error in errors:
            print(f"  - {error}")
        raise ValueError("Invalid configuration. Please check your .env file.")
    
    # Log configuration summary (without secrets)
    print("Configuration loaded:")
    print(f"  - LLM: {LLM_BASE_URL} using {GENERAL_MODEL}")
    print(f"  - Medical Model: {MED_MODEL}")
    print(f"  - Orchestrator: {ORCHESTRATOR_PROVIDER} using {ORCHESTRATOR_MODEL}")
    if OPENMRS_FHIR_BASE_URL:
        print(f"  - FHIR: Connected to {OPENMRS_FHIR_BASE_URL}")
    if FHIR_PARQUET_DIR:
        print(f"  - Local Data: {FHIR_PARQUET_DIR}")
    print(f"  - Environment: {ENV}")

# Run validation on import
if __name__ != "__main__":  # Only validate when imported as module
    try:
        validate_config()
    except ValueError as e:
        print(f"Warning: {e}")