# Multi-Model Chat

Fast, lightweight chat application featuring **Gemma 3 1B** (general chat) and **MedGemma 4B** (medical Q&A), optimized for Apple Silicon Macs.

## Quick Start

**Requirements**: macOS with Apple Silicon (M1/M2/M3), Python 3.12, 16GB+ RAM, 25GB disk space

```bash
# Install Python if needed
brew install python@3.12

# Clone and run
git clone <your-repo-url>
cd medgemma_chat
./start_server.sh
```

The script handles everything automatically: Poetry installation, dependencies, model downloads (~25GB), and server startup.

### First Run
1. **Hugging Face Login**: You'll be prompted to authenticate (required for MedGemma)
   - Visit [MedGemma page](https://huggingface.co/google/medgemma-4b-it) and accept terms
   - Get token from [settings](https://huggingface.co/settings/tokens)
   - Paste when prompted
2. **Model Download**: ~25GB total (one-time, cached locally)
3. **Server Starts**: http://127.0.0.1:3000

## Usage

1. Open `client/index.html` in your browser
2. Select a model:
   - **📱 Gemma 3 1B**: Ultra-fast general chat (1-3s responses, English-only)
   - **🏥 MedGemma**: Medical Q&A with disclaimers (3-8s responses)
3. Choose a system prompt preset or write your own
4. Start chatting!

### Example Prompts
- General: "Explain quantum computing in simple terms"
- Medical: "What are the symptoms of dehydration?"
- Custom: "Act as a Python tutor and explain list comprehensions"

## Performance

Optimized for Apple Silicon with:
- Metal Performance Shaders (MPS) acceleration
- bfloat16 precision for speed and stability
- Smart memory management
- Context-aware token limits

**Expected response times** (M2/M3 MacBook):
- Gemma 3 1B: 1-3 seconds
- MedGemma: 3-8 seconds

## Configuration

### Switch Models

Edit `MODEL_CONFIG` in `server/main.py`:

```python
"general": {
    "model_id": "google/gemma-2-2b-it",  # Change to any model
    "display_name": "Gemma 2 2B",
    # ...
}
```

**Popular alternatives**:
- `google/gemma-2-2b-it` - Better performance, 2B params
- `microsoft/Phi-3-mini-4k-instruct` - Microsoft's 3.8B model
- `mistralai/Mistral-7B-Instruct-v0.2` - Powerful 7B model

### Add New Models

```python
"coding": {
    "enabled": True,
    "model_id": "google/codegemma-2b",
    "display_name": "CodeGemma",
    "icon": "💻",
    # ...
}
```

## Memory Management

Models use ~20GB RAM when loaded. Memory is managed automatically by PyTorch/Python for optimal performance. 

If you experience memory issues:

```bash
# Check memory usage
curl http://127.0.0.1:3000/health | jq .memory

# Manual cleanup (only when needed)
curl -X POST http://127.0.0.1:3000/memory/cleanup

# Or just restart the server
```

**Note**: Automatic cleanup was removed in v1.1 as it was causing 30-60 second delays.

## Troubleshooting

### Setup Issues

**Python not found**:
```bash
brew install python@3.12
python3.12 --version
```

**Poetry issues**:
```bash
# Manual install
curl -sSL https://install.python-poetry.org | python3.12 -
export PATH="$HOME/.local/bin:$PATH"
```

### Performance Issues

- **Slow responses**: Check Activity Monitor for memory pressure
- **Very slow (>60s)**: Fixed in v1.1 - was caused by aggressive memory cleanup
- **First response slow**: Normal - models warming up
- **Gemma 3 1B**: Optimized for 1-3 second responses with greedy decoding

### Common Errors

- **Connection refused**: Server not running, check terminal
- **Out of memory**: Close other apps, need 16GB+ free
- **Authentication failed**: Accept MedGemma license on Hugging Face

## Advanced

### Manual Setup
```bash
poetry install
poetry run python server/main.py
```

### Performance Packages
```bash
poetry install --extras performance
```

### System Prompts
The app uses a 4-layer prompt system:
1. Model defaults (built-in)
2. User selection (presets/custom)
3. Response optimization (automatic)
4. Smart token management

See [prompt optimization research](https://medium.com/data-science-in-your-pocket/claudes-system-prompt-explained-d9b7989c38a3) for details.