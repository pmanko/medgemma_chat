#!/bin/bash

# Multi-Model Chat Server Startup Script
# Optimized for Apple Silicon (M2/M3 MacBook Pro)
# One-step setup assuming Python 3.12 is available

echo "🚀 Starting Multi-Model Chat Server..."
echo "📍 This may take 5-10 minutes on first run (model downloads)"
echo ""

# Function to find Python 3.12
find_python312() {
    # Try different common locations for Python 3.12
    for python_cmd in python3.12 /opt/homebrew/bin/python3.12 /usr/local/bin/python3.12 python3; do
        if command -v "$python_cmd" &> /dev/null; then
            local version=$("$python_cmd" --version 2>&1 | grep -o '3\.[0-9][0-9]*')
            if [[ "$version" == "3.12" ]] || [[ "$version" == "3.11" ]] || [[ "$version" == "3.10" ]] || [[ "$version" == "3.9" ]]; then
                echo "$python_cmd"
                return 0
            fi
        fi
    done
    return 1
}

# Check for compatible Python version
echo "🐍 Checking Python version..."
PYTHON_CMD=$(find_python312)
if [ $? -ne 0 ]; then
    echo "❌ Python 3.9-3.12 not found. Please install Python 3.12:"
    echo "   brew install python@3.12"
    echo ""
    echo "Available Python versions:"
    ls -la /opt/homebrew/bin/python* 2>/dev/null || ls -la /usr/local/bin/python* 2>/dev/null || echo "No Python installations found in common locations"
    exit 1
fi

echo "✅ Found compatible Python: $PYTHON_CMD ($($PYTHON_CMD --version))"

# Function to install Poetry
install_poetry() {
    echo "📦 Installing Poetry with $PYTHON_CMD..."
    
    # Install Poetry using the correct Python version
    curl -sSL https://install.python-poetry.org | $PYTHON_CMD -
    
    # Add Poetry to PATH for current session
    export PATH="$HOME/.local/bin:$PATH"
    
    # Add to shell profile if not already there
    SHELL_PROFILE=""
    if [[ "$SHELL" == *"zsh"* ]]; then
        SHELL_PROFILE="$HOME/.zshrc"
    elif [[ "$SHELL" == *"bash"* ]]; then
        SHELL_PROFILE="$HOME/.bash_profile"
    fi
    
    if [[ -n "$SHELL_PROFILE" ]]; then
        if ! grep -q 'export PATH="$HOME/.local/bin:$PATH"' "$SHELL_PROFILE" 2>/dev/null; then
            echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$SHELL_PROFILE"
            echo "✅ Added Poetry to PATH in $SHELL_PROFILE"
        fi
    fi
    
    # Verify Poetry installation
    if command -v poetry &> /dev/null; then
        echo "✅ Poetry installed successfully: $(poetry --version)"
        return 0
    else
        echo "❌ Poetry installation failed"
        return 1
    fi
}

# Check if Poetry is installed
if ! command -v poetry &> /dev/null; then
    echo "📦 Poetry not found. Installing automatically..."
    if ! install_poetry; then
        echo "❌ Failed to install Poetry. Please install manually:"
        echo "   curl -sSL https://install.python-poetry.org | $PYTHON_CMD -"
        echo "   export PATH=\"\$HOME/.local/bin:\$PATH\""
        exit 1
    fi
fi

# Ensure Poetry is using the correct Python version
echo "🔧 Configuring Poetry to use $PYTHON_CMD..."
if ! poetry env use "$PYTHON_CMD" 2>/dev/null; then
    echo "⚠️  Warning: Could not set Poetry Python version, but continuing..."
fi

# Verify Poetry is working with correct Python
POETRY_PYTHON=$(poetry run python --version 2>/dev/null)
if [[ -n "$POETRY_PYTHON" ]]; then
    echo "✅ Poetry using: $POETRY_PYTHON"
else
    echo "⚠️  Warning: Could not verify Poetry Python version"
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