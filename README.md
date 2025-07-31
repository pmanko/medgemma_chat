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

## System Prompt Architecture

The application uses an intelligent **4-layer system prompt system** for optimal model behavior:

### How System Prompts Work Together

**1. Model Defaults (Built-in)**
- **Phi-3**: General-purpose conversational AI
- **MedGemma**: Medical-focused responses with evidence-based information

**2. User Selection (Customizable)**
- 💬 **Default**: Uses model-optimized defaults only
- 🤝 **Helpful**: Emphasizes helpfulness and honesty
- 🏥 **Medical**: Educational medical information (not advice)  
- 🔬 **Researcher**: Evidence-based, well-researched responses
- ✏️ **Custom**: Write your own system prompt

**3. Response Optimization (Automatic)**
- Ensures markdown formatting for better readability
- Prevents response cutoffs mid-sentence
- Includes appropriate disclaimers for medical content
- Optimizes response length (400-800 words for medical topics)

**Final Prompt Structure:**
```
[Model-specific defaults] + [Your customization] + [Response guidance]
```

**Smart Token Management:**
- Automatically manages token limits (800 for MedGemma, 1000 for Phi-3)
- Prioritizes response guidance (never truncated)
- Gracefully handles long custom prompts
- Based on [prompt optimization research](https://medium.com/data-science-in-your-pocket/claudes-system-prompt-explained-d9b7989c38a3)

## Open Client
Open `client/index.html` in your browser or run:
```bash
cd client && python3 -m http.server 8000
# Then visit: http://localhost:8000
```

## Usage

1. **Choose Model**: Select **Phi-3** (general) or **MedGemma** (medical)
2. **Set System Prompt**: Pick from presets or write custom instructions
3. **Ask Questions**: Type your question and send

**Examples:**
- **Phi-3 + Researcher**: "Explain quantum computing with recent research"
- **MedGemma + Medical**: "What are the diagnostic criteria for diabetes?"
- **Phi-3 + Custom**: "Act as a Python tutor and explain functions"

## Performance Optimization

The system includes several performance optimizations for Apple Silicon:

### **Automatic Optimizations**
- ✅ **Metal Performance Shaders (MPS)** - GPU acceleration on Apple Silicon
- ✅ **Eager Attention** - Optimized for Apple Silicon and Gemma model compatibility
- ✅ **Fast Image Processor** - Optimized medical image processing for MedGemma
- ✅ **bfloat16 precision** - Better numerical stability + speed
- ✅ **KV Caching** - Reuse computations across requests
- ✅ **Optimized generation parameters** - Tuned for speed vs quality

### **Performance Tips**

**For Best Performance:**
1. **Close other apps** - Free up RAM for models (need 16GB+)
2. **Use shorter prompts** - Faster processing, less context
3. **Clear history periodically** - Reduces context length overhead
4. **Keep system cool** - Thermal throttling affects performance

**Expected Performance (M2/M3 MacBook Pro):**
- **Phi-3**: ~2-5 seconds per response
- **MedGemma**: ~3-8 seconds per response  
- **First response**: Slower due to model warmup

### **Install Performance Packages** (Optional)

For maximum performance, install additional optimization packages:

```bash
# After first successful run, add optional performance packages
poetry install --extras performance

# Performance extras include:
# - optimum: Hugging Face performance optimizations  
# - bitsandbytes: Memory-efficient quantization (optional)
```

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

### Performance Issues

**Slow responses (>10 seconds)**:
```bash
# Check system resources
Activity Monitor → Memory tab → Memory Pressure should be green

# Free up RAM
sudo purge  # Clear system caches

# Check thermal throttling  
sudo powermetrics --samplers smc -n 1 | grep -i temp
```

**Models not using MPS (GPU)**:  
- Check: Server logs should show "✅ MPS Available"
- If not: Update PyTorch: `poetry add torch --latest`

**Attention implementation**:
- Uses **eager attention** by default (recommended for Gemma models)
- Optimized for Apple Silicon MPS
- No additional installation needed

### Runtime Issues

**"Connection refused" errors**: Server not running - check terminal for errors

**Out of memory**: Close other apps, ensure 16GB+ RAM available

**Model download fails**: Check internet connection and disk space

**Authentication issues**: Ensure you accepted MedGemma license terms

**Very slow first response**: Models loading into memory (normal)
