# Installation Guide

Complete setup instructions for the Medical Multi-Agent Chat System using the A2A SDK.

## System Requirements

### Minimum Requirements
- **Python**: 3.10 or higher (A2A SDK requirement)
- **Memory**: 8GB RAM
- **Disk**: 10GB free space
- **OS**: Windows 10+, macOS 10.15+, Ubuntu 20.04+

### Recommended Requirements
- **Python**: 3.11+ for best performance
- **Memory**: 16GB+ RAM (for running local LLMs)
- **GPU**: NVIDIA/AMD/Apple Silicon (for acceleration)
- **Disk**: 20GB+ (for multiple models)

## Installation Methods

### Method 1: Quick Install (Recommended)

```bash
# Clone the repository
git clone <repository-url>
cd projects/multiagent_chat

# Install with Poetry (recommended)
poetry install

# This installs all dependencies defined in pyproject.toml
```

### Method 2: Development Install

```bash
# Clone repository
git clone <repository-url>
cd projects/multiagent_chat

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install with development dependencies
poetry install --with dev

# Or just the base dependencies
poetry install --only main
```

### Method 3: Docker Install

```bash
# Build Docker images
docker build -t multiagent-chat .

# Or use docker-compose
docker-compose build
```

See [Docker Deployment Guide](../deployment/docker.md) for details.

## Core Dependencies

### Required Packages

```toml
# pyproject.toml key dependencies
python = ">=3.10,<3.13"
a2a-sdk = "^0.3.0"           # Official A2A SDK
fastapi = "^0.104.1"         # Web framework
uvicorn = "^0.24.0"          # ASGI server
httpx = "^0.25.0"            # Async HTTP client
pydantic = "^2.0.0"          # Data validation
python-dotenv = "^1.0.1"     # Environment management
```

### Optional Dependencies

```toml
# For FHIR/SQL integration
PyHive = "^0.7.0"           # Spark SQL connector
thrift = "^0.20.0"          # Thrift protocol
duckdb = "^0.8.0"           # Local SQL engine

# Development tools
pytest = "^7.4.0"           # Testing
black = "^23.0.0"           # Code formatting
isort = "^5.12.0"           # Import sorting
```

## Installing the A2A SDK

The A2A SDK is the core dependency for agent communication:

### From PyPI (Stable)
```bash
poetry add a2a-sdk
```

### From Source (Latest)
```bash
# Clone A2A SDK
git clone https://github.com/a2aproject/a2a-python.git
cd a2a-python

# Install in development mode
poetry install
```

### Verify Installation
```python
# Test A2A SDK installation
python -c "import a2a; print(f'A2A SDK {a2a.__version__} installed')"
```

## Setting Up LLM Providers

The agents require an LLM backend. Choose one:

### LM Studio (Recommended)

See [LM Studio Setup Guide](lm-studio.md) for detailed instructions.

**Quick setup:**
1. Download LM Studio from [lmstudio.ai](https://lmstudio.ai)
2. Download models (e.g., Llama 3, MedGemma)
3. Start server (default port 1234)
4. Set `LLM_BASE_URL=http://localhost:1234` in `.env`

### Optional: Gemini for Orchestration

You can optionally use Google Gemini for better routing decisions while still using LM Studio for the actual agents:

```env
# Add to .env for Gemini orchestration
ORCHESTRATOR_PROVIDER=gemini
GEMINI_API_KEY=your-api-key  # Get from aistudio.google.com/apikey
ORCHESTRATOR_MODEL=gemini-1.5-flash
```

This hybrid approach gives you:
- Fast, intelligent routing via Gemini
- Private, local processing via LM Studio
- Best of both worlds!

## Environment Configuration

1. **Copy the template**:
```bash
cp env.example .env
```

2. **Minimal configuration** (just 1-2 lines!):
```env
# That's all you need for basic setup!
LLM_BASE_URL=http://localhost:1234
GENERAL_MODEL=llama-3-8b-instruct  # Optional if using this model name
```

3. **Test your configuration**:
```bash
# Test current configuration
poetry run python test_config.py

# Test with minimal config (just LLM_BASE_URL)
poetry run python test_config.py minimal

# Test with Gemini orchestration
poetry run python test_config.py gemini
```

This will verify your settings and check connectivity to LM Studio.

For advanced options like [Gemini orchestration](configuration.md#optional-use-google-gemini-for-orchestration) or [clinical data sources](configuration.md#optional-clinical-data-sources), see the [Configuration Guide](configuration.md).

## Verify Installation

### 1. Check Python Version
```bash
python --version  # Should be 3.10+
```

### 2. Test Dependencies
```python
# test_deps.py
import sys
print(f"Python: {sys.version}")

try:
    import a2a
    print(f"✓ A2A SDK: {a2a.__version__}")
except ImportError:
    print("✗ A2A SDK not installed")

try:
    import fastapi
    print(f"✓ FastAPI: {fastapi.__version__}")
except ImportError:
    print("✗ FastAPI not installed")

try:
    import httpx
    print(f"✓ HTTPX: {httpx.__version__}")
except ImportError:
    print("✗ HTTPX not installed")
```

### 3. Test Agent Launch
```bash
# Dry run to check if agents can start
python launch_a2a_agents.py test
```

## Troubleshooting Installation

### Common Issues

| Problem | Solution |
|---------|----------|
| **Python version error** | Install Python 3.10+ using pyenv or conda |
| **poetry not found** | Install Poetry: `curl -sSL https://install.python-poetry.org | python3 -` |
| **Permission denied** | Poetry manages virtual environments automatically |
| **a2a-sdk not found** | Check PyPI connectivity, try `poetry add a2a-sdk --source pypi` |
| **Import errors** | Clear Poetry cache: `poetry cache clear pypi --all` |
| **SSL certificate error** | Update certificates or use `--trusted-host` |

### Platform-Specific Issues

#### Windows
```powershell
# If scripts disabled
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser

# Long path support
git config --global core.longpaths true
```

#### macOS
```bash
# Install Xcode tools if needed
xcode-select --install

# Fix SSL issues
brew install ca-certificates
```

#### Linux
```bash
# Install Python dev headers
sudo apt-get install python3-dev  # Debian/Ubuntu
sudo yum install python3-devel     # RHEL/CentOS
```

## Next Steps

✅ Installation complete? Continue with:

1. [**Quick Start Guide**](quick-start.md) - Run your first query
2. [**LM Studio Setup**](lm-studio.md) - Configure local LLM
3. [**Configuration Guide**](configuration.md) - Advanced settings

## Getting Help

- **Issues**: [GitHub Issues](https://github.com/...)
- **Discussions**: [GitHub Discussions](https://github.com/...)
- **A2A SDK Docs**: [a2aprotocol.ai](https://a2aprotocol.ai)
