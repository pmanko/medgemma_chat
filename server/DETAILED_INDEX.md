This project is an ambitious endeavor to build a **Multi-Agent Medical System** that leverages advanced AI models and architectures to process and respond to complex clinical queries. The core of its design involves sophisticated use of **Retrieval-Augmented Generation (RAG)**, a **Mixture of Experts** approach, and a progressive adoption of the **Agent2Agent (A2A) Protocol** for inter-agent communication and discovery.

Here is a detailed index of the project, outlining its architecture, components, and key functionalities, along with the relevant sources for context:

### Project Vision and Goal
The overarching goal is to create a **proof-of-concept multi-agent medical chat application** that demonstrates the core principles of the emerging **Agent2Agent (A2A) protocol**. The system is designed to feature a decentralized collection of specialized AI agents that can discover each other and collaborate to answer complex user queries by connecting to both live and local healthcare data sources.

### Architectural Foundation: Simulating an Agent2Agent (A2A) Ecosystem
The project moves away from a hardcoded, centralized orchestrator towards a more dynamic, decentralized model by simulating key A2A concepts within its Python backend. This simulation is a stepping stone to later integrating with native A2A services using the official SDK.

1.  **The Agent Registry (Simulated Discovery)**: This component is a simple, shared Python dictionary or JSON file that acts as the service discovery mechanism. When each agent starts, it registers itself, providing a unique `agent_id`, a human-readable description of its capabilities, and the specific `task_name` it is designed to handle. This allows any agent to look up what other agents are available and what they can do. This mechanism directly mirrors A2A's focus on agent discovery. The project later evolved this into exposing a `/manifest` endpoint that lists agent skills and their input schemas, reflecting A2A's discovery conventions.
2.  **The A2A Message Bus (Simulated Communication)**: This is an in-memory message queue, typically using Python's `queue.Queue`, that all agents connect to. This bus decouples the agents, allowing them to post messages to the bus and listen for messages addressed to them, thereby simulating a peer-to-peer network rather than direct calls. This aligns with A2A's goal of decentralized, peer-to-peer communication.
3.  **The Standardized Message Format**: All internal messages posted to the bus adhere to a consistent JSON structure, serving as the project's internal "protocol" for inter-agent communication. This format includes keys such as `message_id`, `sender_id`, `receiver_id`, `task_name`, `payload` (containing `query` and optional `data`), and `status`. This is analogous to A2A's formal, standardized message format that ensures interoperability.

### The Agents and Their A2A Roles
Each agent in the system is designed as an independent worker process, utilizing a single headless LM Studio server for local LLM "brains" or external API endpoints.

1.  **user_proxy_agent**:
    *   **LLM Model**: Llama 3 or Gemini (configured via `ORCHESTRATOR_PROVIDER` and `ORCHESTRATOR_MODEL`).
    *   **Task Name**: `initiate_query`.
    *   **Description**: This agent listens for initial user queries from the web UI, consults the **Agent Registry** to find the best agent (skill) for the job, and kicks off the first task. It performs "semantic routing" by prompting its LLM to determine which specialist agent is best suited to handle the query based on the registry's contents. It then creates a new message with the appropriate `task_name` and `receiver_id` and posts it to the **Message Bus** for the chosen specialist to pick up. It also handles passing arguments like `scope` to the chosen skill.
2.  **medgemma_agent**:
    *   **LLM Model**: MedGemma.
    *   **Task Name**: `answer_medical_question`.
    *   **Description**: Provides expert answers to general medical questions. It registers its skill with an input schema that includes the `query`.
3.  **clinical_research_agent** (evolved from `openmrs_fhir_agent` and `local_datastore_agent`):
    *   **LLM Model**: Gemma 7B (for query generation) and MedGemma (for clinical synthesis).
    *   **Task Names/Skills**: Initially `query_live_fhir_server` and `query_local_fhir_datastore`. Later consolidated into a single `clinical_research` agent with a single skill `clinical_research` that accepts parameters for different functionalities.
    *   **Description**: This agent is a key focus, designed with **dual-prompt capabilities** to convert natural language questions into either:
        *   **Live FHIR API queries**: For a server like OpenMRS (e.g., `https://fhir.openmrs.org/`). It uses a specialized system prompt for the generalist LLM (Gemma 7B) to generate a FHIR API query string, then executes an HTTP GET call and summarizes the JSON response.
        *   **Spark SQL queries**: For local Parquet-on-FHIR data, leveraging tools like Google's FHIR Data Pipes that convert FHIR resources to Parquet. It uses a different system prompt for the generalist LLM (Gemma 7B) for Text-to-SQL generation against specified table schemas. The execution uses PyHive for Spark Thrift connectivity.
    *   **Scope Control**: The agent incorporates "scope" as a parameter to its `clinical_research` skill, supporting `facility` and `hie` (Health Information Exchange) levels. The orchestrator (user_proxy_agent) determines this scope as part of its skill selection. The agent's internal prompts are scope-aware and enforce constraints (e.g., ensuring facility-specific filters for `facility` scope).
    *   **Clinical Synthesis**: After structured data is retrieved (from either FHIR API or Spark SQL), MedGemma is specifically used for the final clinical synthesis and interpretation, integrating the retrieved data into a comprehensive, human-readable response. This is considered MedGemma's strength, separating it from the query generation task handled by a generalist LLM.

### Backend Components and Functionality
The backend is built using FastAPI and orchestrates the agents and their interactions.

*   **`server/a2a_layer.py`**: Implements the simulated Agent Registry, Message Bus, and utilities for creating standardized A2A messages.
*   **`server/llm_clients.py`**: Provides thin wrappers to call external OpenAI-compatible API endpoints (e.g., LM Studio, custom servers) and the Google Gemini API. This centralizes LLM communication logic and supports configurable endpoints via environment variables.
*   **`server/agents/`**: This directory contains the Python modules for each individual agent (`user_proxy_agent.py`, `medgemma_agent.py`, `fhir_agent.py` for clinical research). Each agent runs in its own thread.
*   **`server/config.py`**: Centralizes environment variable-based settings for LLM base URLs, model names, OpenMRS configuration, Spark Thrift details, and agent timeouts.
*   **`server/schemas.py`**: Defines Pydantic models for request/response bodies, including the `/chat` endpoint's schema (which now includes `scope`, `facility_id`, `org_ids`) and the internal A2A message structure.
*   **`server/main.py`**: The core FastAPI application. On startup, it initializes the Agent Registry and Message Bus, and starts the listener loops for each agent in separate threads. It exposes:
    *   **`POST /chat` endpoint**: This is the primary endpoint for the agentic mode. It takes a user query, wraps it in a standard A2A message, posts it to the **Message Bus** addressed to the `user_proxy_agent`, and then waits for a correlated response from the `web_ui` mailbox.
    *   **`GET /manifest` endpoint**: Returns a JSON output showing the registered agents, their `task_name` (or `agent_id`), `description`, and a detailed list of `skills` with their `input_schema`. This aligns with A2A discovery patterns.
    *   **`POST /generate/general` and `POST /generate/medgemma` endpoints**: These are maintained for backward compatibility, routing requests directly to the configured external general LLM or MedGemma endpoint.
*   **`server/fhir_store.py`**: A simple Python dictionary acting as a mock FHIR patient database for demo purposes, used by the `clinical_research_agent` when no live or Spark connection is configured.

### Frontend (User Interface)
The project utilizes a simple web-based chat interface.

*   **UI Framework**: Streamlit (initially proposed) or a custom HTML/CSS/JavaScript interface (as implemented).
*   **Modes**: The UI features a mode switch (a dropdown) to select between two interaction modes:
    *   **"🧠 Direct (per-model)"**: This mode sends queries directly to the `/generate/general` or `/generate/medgemma` endpoints, allowing the user to explicitly choose which model to interact with.
    *   **"🕸️ Agents (A2A)"**: This mode sends queries to the `/chat` endpoint, allowing the **user_proxy_agent** to perform semantic routing and dispatch the query to the appropriate specialist agent.

### Key Architectural Patterns and Concepts
The project’s design embodies several advanced AI architecture patterns:

1.  **Retrieval-Augmented Generation (RAG)**: The entire system, especially the `clinical_research_agent`, is a sophisticated form of RAG.
    *   **Retrieve-then-Generate Principle**: The core idea is to augment the LLM's knowledge by retrieving relevant information from an external source *before* generating a response.
    *   **Structured FHIR RAG**: Unlike classic text-based RAG that relies on semantic search over unstructured text chunks, this project's RAG specializes in "query generation" (Text-to-SQL or Text-to-FHIR API) against structured databases (FHIR APIs, Parquet files) to retrieve precise, structured JSON objects or table rows.
    *   **Just-in-Time Context Injection**: Retrieved structured data is dynamically formatted and injected directly into the LLM's prompt, often within clear XML-like tags (e.g., `<clinical_context>...<clinical_context>`), for the final synthesis.
    *   **Hybrid RAG (Future Phase 2)**: A proposed enhancement involves integrating semantic search using embeddings for unstructured clinical notes within FHIR data. This would involve:
        *   **Offline Embedding Pipeline**: Extracting text from FHIR fields (e.g., notes, conclusions), chunking it, converting chunks to vector embeddings using a **specialized text-embedding model** (like a PubMedBERT successor), and storing them in a vector database (e.g., ChromaDB, FAISS).
        *   **Online Hybrid Retrieval Flow**: At query time, perform semantic search on the vector database using the user's query embedding to find relevant resource IDs for unstructured notes. Then, perform precise SQL queries to pull the *full, structured FHIR data* for those IDs. Finally, inject this combined data into the MedGemma prompt for synthesis.
2.  **Mixture of Experts**: The project explicitly uses different LLMs for different sub-tasks based on their strengths:
    *   **Generalist LLMs (Gemma 7B, Llama 3 8B)**: Used for logical and code-generation tasks, specifically for **query generation** (Text-to-SQL or Text-to-FHIR API calls) within the `clinical_research_agent` and for **semantic routing** in the `user_proxy_agent`.
    *   **MedGemma**: A **foundational generative multimodal model** optimized for medical applications. Its primary role is **clinical synthesis and generation**, interpreting retrieved structured clinical data and providing a clinically nuanced, accurate, and fluent natural language answer. It uses an internal **MedSigLIP** for vision encoding, but MedSigLIP is *not* used for text-only semantic search. MedGemma's internal text encoder is optimized for text generation, not retrieval.
    *   **Specialized Text-Embedding Models**: Proposed for the future Hybrid RAG's text-only semantic search component (e.g., a successor to PubMedBERT). These models are explicitly trained for **retrieval** and creating highly accurate semantic embeddings for text.
3.  **Model Context Protocol (MCP)**: While not explicitly called out as a separate component in the code, the project's design embodies MCP principles. MCP focuses on how individual agents connect to their **tools, APIs, and resources** with structured inputs and outputs. The `clinical_research_agent`'s workflow of structured query generation and context injection for synthesis is a direct application of an agent accessing its "tools" (FHIR API, Spark DB, MedGemma) through structured inputs and outputs, which aligns with MCP's core.
4.  **Orchestration and Semantic Routing**: The `user_proxy_agent` acts as the orchestrator, using its LLM to intelligently analyze user queries and descriptions of available agents/skills from the **Agent Registry** to determine the best agent for the task.
    *   **Skill-Based Interactions with Typed Inputs**: A refined A2A pattern is adopted where agent capabilities are modeled as "skills" (e.g., `clinical_research`) with clearly defined input schemas (e.g., including `scope`, `facility_id`, `org_ids`). This allows the orchestrator to select a specific skill and provide it with typed arguments, making interactions precise and discoverable.
5.  **LM Studio Integration**: LM Studio is used as the local backend for serving models. Initially, multiple instances were proposed, but this was streamlined to a **single LM Studio headless server** with on-demand model loading. Models are referenced by their API identifiers (e.g., `meta-llama/Llama-3-8B-Instruct-GGUF`). The server is started with `lms server start --headless`.
6.  **Orchestrator Options**: The system supports flexible choice for the main orchestrator (user_proxy_agent's LLM):
    *   **Local Orchestrator**: Uses Llama 3 8B Instruct (a strong open-source option for consumer-grade hardware).
    *   **API Orchestrator**: Uses Google Gemini 1.5 Pro (via API, for native function calling and large context windows) or Anthropic Claude 3 family (Opus, Sonnet, Haiku, also via API).

### Data Sources
The system is designed to interact with two primary types of FHIR data sources:

1.  **Live OpenMRS FHIR Server**: `https://fhir.openmrs.org/`. This public server implements the FHIR R4 standard and supports resources like Patient, Observation, Medication, etc.. The `clinical_research_agent` generates FHIR API queries to retrieve data from this endpoint.
2.  **Google Open Health Stack FHIR Data Pipes (Spark-based FHIR Repository)**: This refers to a framework (`https://github.com/google/fhir-data-pipes`) that facilitates converting FHIR resources to a Parquet-on-FHIR schema, making it easier for analytics. The `clinical_research_agent` can query this local data store by generating Spark SQL queries (via PyHive/Hive Thrift) against the Parquet files.
3.  **FHIR Info Gateway**: This is a related Google Open Health Stack component that acts as a reverse proxy to enforce role-based access control (RBAC) policies on FHIR data, ensuring patient data privacy.

### Evolution and Refinements Throughout the Project
The project has undergone significant architectural refinements based on discussions and best practices:

*   **LM Studio Instances**: Moved from running multiple LM Studio instances on separate ports to a single headless LM Studio server with on-demand model loading, simplifying resource management.
*   **Gemma Model Usage**: Consolidated the "generalist" and "patient context agent" roles to a single Gemma model instance using different system prompts, optimizing resource usage.
*   **FHIR Data Access**: Evolved the `local_datastore_agent` to `clinical_research_agent` capable of both live FHIR API queries and Spark SQL on Parquet-on-FHIR data.
*   **SQL Backend for Parquet**: Switched from DuckDB to Spark Thrift (PyHive) as the preferred client for querying local Parquet-on-FHIR files.
*   **LLM Role in Synthesis**: Refined the RAG workflow to ensure MedGemma (the specialized clinical LLM) performs the *final synthesis* of retrieved data, while generalist LLMs handle query generation and routing.
*   **A2A Protocol Adoption**: Progressed from a custom, internal orchestration pattern to a simulated A2A ecosystem (with Agent Registry, Message Bus, Standardized Message Format). The plan includes preparatory hooks for full **native A2A services** using the official SDK, including exposing `/manifest` endpoints for agent discovery.
*   **Scope Control Integration**: Implemented `facility` and `hie` scope controls within the `clinical_research_agent`, with the orchestrator selecting the scope as a parameterized skill argument in an A2A-compliant manner.

### Index of Sources Provided

*   **Excerpts from "A guide to Multiagent workflows with LM Studio"**: Details model selection strategy (Mistral, Gemma, Llama 3), orchestrator choices (local vs. Gemini API, Claude), LM Studio configuration (multi-instance vs. single headless server), the concept of Mixture of Experts, and initial RAG workflow.
*   **Excerpts from "Agent Communication and Capability Protocols in AI Systems"**: Defines A2A and MCP, their complementarity, and how the project leverages simulated A2A concepts (Registry, Message Bus, Message Format) and MCP-like principles (structured data retrieval, context injection).
*   **Excerpts from "Agent2Agent (A2A) Protocol"**: Core definition of A2A, its importance (interoperability, complex workflows, security), and its relationship with MCP.
*   **Excerpts from "Agent2Agent Protocol: Foundations for Interoperable AI Systems"**: Further elaborates on A2A core concepts (decentralized, standardized communication, discovery, message bus, semantic routing, skill-based interactions), and details the project's evolution towards A2A.
*   **Excerpts from "Decentralized Medical Chat with A2A Agents and FHIR Queries"**: A concise summary of the project's refactoring into a decentralized multi-agent system using simulated A2A, highlighting key agents and FHIR query capabilities.
*   **Excerpts from "FHIR Analytics | Open Health Stack | Google for Developers"**: Provides context on FHIR Data Pipes, Parquet-on-FHIR schema, and tools for FHIR analytics.
*   **Excerpts from "FHIR Info Gateway | Open Health Stack | Google for Developers"**: Explains FHIR Info Gateway as a reverse proxy for role-based access control (RBAC) on FHIR data.
*   **Excerpts from "Health AI Developer Foundations Overview | Google for Developers"**: Overview of Google's Health AI Developer Foundations (HAI-DEF), including open-weight models specialized for health domains like MedGemma and MedSigLIP.
*   **Excerpts from "Home - OpenMRS Core FHIR Implementation Guide v0.1.0"**: Details the OpenMRS FHIR2 module, its commitment to FHIR implementation, and supported FHIR resources.
*   **Excerpts from "MedGemma model card | Health AI Developer Foundations | Google for Developers"**: Detailed information about MedGemma variants (4B, 27B text-only, 27B multimodal), their training data (medical text, images, FHIR-based EHR data), use cases, performance benchmarks, and its relationship with MedSigLIP. This source clarifies that MedSigLIP is the image encoder component of MedGemma, and highlights the distinction between generative models' internal encoders and dedicated retrieval encoders for RAG.
*   **Excerpts from "cursor_refactor_project_for_a2a_medical.md"**: This document, generated by the AI, provides a comprehensive blueprint and detailed step-by-step implementation plan for the A2A-enabled refactor, including file-level changes, configuration, execution flow, and specific agent logic. This document is a primary source for the project's evolution and detailed implementation plan.
*   **Excerpts from "google/appoint-ready at main"**: Provides a basic example of a Hugging Face Space setup with a `medgemma.py` file, indicating a potential avenue for deploying a demo.

This comprehensive index provides a detailed understanding of the project's architecture, components, and the strategic use of various AI protocols and models.

A next step could be to create a specific data model or ERD (Entity-Relationship Diagram) for the FHIR data within the local Parquet files, to ensure the SQL queries generated by the `clinical_research_agent` are accurate and efficient for the intended data schema.