# Multi-Model Chat

Chat application with **Phi-3** (general) and **MedGemma** (medical) models, optimized for Apple Silicon.

## Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3 MacBook)
- **Python 3.9-3.12** (not 3.13 - has compatibility issues)
- **16GB+ RAM** (models require significant memory)
- **25GB+ free disk space** (for model downloads)
- **Hugging Face account** (free - needed for MedGemma access)

## Setup

### 1. Install Python (if needed)
```bash
# Install via Homebrew (recommended)
brew install python@3.12
```

### 2. Install Poetry
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

### 3. Clone & Run
```bash
git clone <your-repo-url>
cd medgemma_chat
./start_server.sh
```

### 4. Authenticate (first run only)
- Script will prompt you to login to Hugging Face
- Visit: https://huggingface.co/google/medgemma-4b-it
- Accept the Health AI Developer Foundation terms
- Get your token from: https://huggingface.co/settings/tokens
- Paste token when prompted

### 5. Open Client
Open `client/index.html` in your browser or run:
```bash
cd client && python3 -m http.server 3000
# Then visit: http://localhost:3000
```

**⏱️ First startup takes 5-10 minutes** (downloads ~25GB of models)

## Usage

Select **Phi-3** for general questions or **MedGemma** for medical questions, then type and send.

**Examples:**
- Phi-3: "Explain quantum computing"
- MedGemma: "What are symptoms of diabetes?"

## Troubleshooting

**"Connection refused" errors**: Server not running - check terminal for errors

**Python 3.13 issues**: Use Python 3.12 instead: `poetry env use python3.12`

**Out of memory**: Close other apps, ensure 16GB+ RAM available

**Model download fails**: Check internet connection and disk space

**Authentication issues**: Ensure you accepted MedGemma license terms
