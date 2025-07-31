# Multi-Model Chat

Chat application with **Phi-3** (general) and **MedGemma** (medical) models, optimized for Apple Silicon.

## Quick Start

1. **Install Poetry**: `curl -sSL https://install.python-poetry.org | python3 -`
2. **Setup & Run**: `./start_server.sh` (handles everything automatically)
3. **Authenticate**: Follow prompts to access MedGemma (one-time setup)
4. **Open**: `client/index.html` in browser

Requires 16GB+ RAM and Apple Silicon Mac.

## Usage

Select **Phi-3** for general questions or **MedGemma** for medical questions, then type and send.

**Examples:**
- Phi-3: "Explain quantum computing"
- MedGemma: "What are symptoms of diabetes?"
