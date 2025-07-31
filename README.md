# Multi-Model Chat

Chat application with **Phi-3** (general) and **MedGemma** (medical) models, optimized for Apple Silicon.

## One-Step Setup

**Prerequisites**: macOS with Apple Silicon + Python 3.12

```bash
# 1. Install Python 3.12 (if needed)
brew install python@3.12

# 2. Clone and run (everything else is automatic!)
git clone <your-repo-url>
cd medgemma_chat
./start_server.sh
```

**That's it!** The script automatically:
- ✅ Finds your Python 3.12 installation
- ✅ Installs Poetry with correct Python version  
- ✅ Manages all dependencies
- ✅ Handles Hugging Face authentication
- ✅ Downloads models (25GB - be patient!)
- ✅ Starts the server

### What You Need

- **macOS** with Apple Silicon (M1/M2/M3 MacBook)
- **Python 3.12** (`brew install python@3.12`) 
- **16GB+ RAM** (models require significant memory)
- **25GB+ free disk space** (for model downloads)
- **Hugging Face account** (free - script will help you set up)

### First Run Process

When you run `./start_server.sh` for the first time:

#### 1. Authentication (Required for MedGemma)
- Script will prompt you to login to Hugging Face
- Visit: https://huggingface.co/google/medgemma-4b-it  
- Accept the Health AI Developer Foundation terms
- Get your token from: https://huggingface.co/settings/tokens
- Paste token when prompted

#### 2. Model Download (~25GB)
- Phi-3: ~7GB download
- MedGemma: ~18GB download
- Models cached locally after first download

#### 3. Server Starts
- FastAPI server runs on `http://127.0.0.1:3000`
- Ready to accept chat requests!

## Open Client
Open `client/index.html` in your browser or run:
```bash
cd client && python3 -m http.server 8000
# Then visit: http://localhost:8000
```

## Usage

Select **Phi-3** for general questions or **MedGemma** for medical questions, then type and send.

**Examples:**
- Phi-3: "Explain quantum computing" 
- MedGemma: "What are symptoms of diabetes?"

## Manual Poetry Setup (Advanced Users Only)

If the automatic setup doesn't work, you can install Poetry manually:

### Option A: Install with specific Python version
```bash
curl -sSL https://install.python-poetry.org | python3.12 -
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

### Option B: Install via pipx
```bash
brew install pipx
pipx ensurepath
pipx install poetry
```

### Verify Installation
```bash
poetry --version
poetry env info
poetry env use python3.12  # If needed
```

## Troubleshooting

### Automatic Setup Issues

**"Python 3.9-3.12 not found"**:
```bash
# Install Python 3.12 via Homebrew
brew install python@3.12

# Verify installation
python3.12 --version
```

**"Poetry installation failed"**:
```bash
# Try manual installation
curl -sSL https://install.python-poetry.org | python3.12 -
export PATH="$HOME/.local/bin:$PATH"

# Then restart the script
./start_server.sh
```

**Script can't find Python**:
```bash
# Check where Python is installed
which python3.12
which python3

# If using Homebrew Python, it should be in:
ls -la /opt/homebrew/bin/python*
```

### Poetry/Python Issues (Manual Setup)

**"Poetry not found"**: 
- Restart terminal after installation
- Check: `echo $PATH` should include `$HOME/.local/bin`
- Manually add: `export PATH="$HOME/.local/bin:$PATH"`

**Wrong Python version in Poetry**:
```bash
# Check current Python
poetry env info

# Set correct Python (try these in order)
poetry env use python3.12
poetry env use /opt/homebrew/bin/python3.12
poetry env use /usr/local/bin/python3.12

# Force recreate environment
poetry env remove python
poetry install
```

**"pyproject.toml changed significantly"**:
```bash
poetry lock --no-update
poetry install
```

**Python 3.13 compatibility issues**:
- Install Python 3.12: `brew install python@3.12`
- Set Poetry to use it: `poetry env use python3.12`
- NumPy/PyTorch don't fully support 3.13 yet

### Runtime Issues

**"Connection refused" errors**: Server not running - check terminal for errors

**Out of memory**: Close other apps, ensure 16GB+ RAM available

**Model download fails**: Check internet connection and disk space

**Authentication issues**: Ensure you accepted MedGemma license terms
