#!/bin/bash

# Multi-Model Chat Server Startup Script
# Optimized for Apple Silicon (M2/M3 MacBook Pro)

echo "🚀 Starting Multi-Model Chat Server..."
echo "📍 This may take 5-10 minutes on first run (model downloads)"
echo ""

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "❌ Poetry not found. Please install Poetry first:"
    echo "   curl -sSL https://install.python-poetry.org | python3 -"
    echo "   Or: pip install poetry"
    exit 1
fi

# Check if pyproject.toml exists
if [ ! -f "pyproject.toml" ]; then
    echo "❌ pyproject.toml not found. Are you in the right directory?"
    exit 1
fi

# Handle lock file and dependencies
echo "📦 Setting up dependencies..."

# Check if lock file exists or is outdated
if [ ! -f "poetry.lock" ] || ! poetry check --lock 2>/dev/null; then
    echo "🔄 Updating lock file..."
    poetry lock
fi

# Install dependencies
echo "📦 Installing dependencies..."
poetry install

# Check if dependencies are working
if ! poetry run python -c "import fastapi, torch, transformers" 2>/dev/null; then
    echo "❌ Dependencies not working properly. Try:"
    echo "   poetry install --no-cache"
    exit 1
fi

# Check Hugging Face authentication for MedGemma access
echo "🔐 Checking Hugging Face authentication..."
HF_USER=$(poetry run huggingface-cli whoami 2>/dev/null)
if [ -z "$HF_USER" ] || [[ "$HF_USER" == *"not logged in"* ]]; then
    echo ""
    echo "❌ Not logged in to Hugging Face. MedGemma requires authentication."
    echo ""
    echo "📋 To get access:"
    echo "   1. Go to: https://huggingface.co/google/medgemma-4b-it"
    echo "   2. Create/Login to your Hugging Face account"
    echo "   3. Accept the Health AI Developer Foundation terms"
    echo "   4. Get your token from: https://huggingface.co/settings/tokens"
    echo "   5. Create a 'Read' token and copy it"
    echo ""
    echo "🔑 Ready to login? (y/n)"
    read -r response
    if [[ "$response" =~ ^([yY][eE][sS]|[yY])$ ]]; then
        echo "Running login command..."
        poetry run huggingface-cli login
        echo ""
        # Verify login worked
        HF_USER=$(poetry run huggingface-cli whoami 2>/dev/null)
        if [ -z "$HF_USER" ] || [[ "$HF_USER" == *"not logged in"* ]]; then
            echo "❌ Login failed. Please try again or check your token."
            exit 1
        fi
        echo "✅ Successfully authenticated as: $HF_USER"
    else
        echo "❌ Authentication required for MedGemma. Exiting..."
        exit 1
    fi
else
    echo "✅ Already authenticated as: $HF_USER"
fi

# Check MPS availability
echo "🔍 Checking Apple Silicon MPS availability..."
if poetry run python -c "import torch; print('✅ MPS Available' if torch.backends.mps.is_available() else '⚠️  MPS Not Available - Using CPU fallback')" 2>/dev/null; then
    echo ""
else
    echo "❌ PyTorch not properly installed"
    exit 1
fi

# Start the server
echo "📡 Starting FastAPI server on http://127.0.0.1:3000"
echo "📱 Open client/index.html in your browser to start chatting"
echo ""
echo "💡 Tip: First startup will be slower due to model loading..."
echo ""

# Run server on port 3000
poetry run uvicorn server.main:app --port 3000 --reload