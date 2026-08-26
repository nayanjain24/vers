#!/usr/bin/env bash
# ==============================================================================
# VERS v5.0 — One-Click Automated Startup Script
# Works on macOS, Linux, and GitHub Codespaces
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "========================================================"
echo "  🚀 Starting VERS v5.0 Visual Emergency Response System"
echo "========================================================"

# Detect Python interpreter
if [ -d "$PROJECT_ROOT/.venv" ]; then
    PYTHON_EXEC="$PROJECT_ROOT/.venv/bin/python"
elif [ -n "$VIRTUAL_ENV" ]; then
    PYTHON_EXEC="$VIRTUAL_ENV/bin/python"
elif command -v python3 &>/dev/null; then
    PYTHON_EXEC="$(command -v python3)"
else
    PYTHON_EXEC="python"
fi

# If virtual environment doesn't exist, create it
if [ ! -f "$PYTHON_EXEC" ]; then
    echo "📦 Creating Python virtual environment in .venv..."
    python3 -m venv .venv
    PYTHON_EXEC="$PROJECT_ROOT/.venv/bin/python"
fi

# Ensure .env exists
if [ ! -f "$PROJECT_ROOT/.env" ] && [ -f "$PROJECT_ROOT/.env.example" ]; then
    echo "📄 Initializing .env from .env.example..."
    cp "$PROJECT_ROOT/.env.example" "$PROJECT_ROOT/.env"
fi

# Install dependencies if --install or --install-only is passed or on fresh setup
if [ "$1" == "--install" ] || [ "$1" == "--install-only" ] || [ ! -d "$PROJECT_ROOT/frontend/node_modules" ]; then
    echo "📦 Installing Python & Node dependencies..."
    "$PYTHON_EXEC" -m pip install --upgrade pip
    "$PYTHON_EXEC" -m pip install mediapipe==0.10.14 protobuf==4.25.9
    "$PYTHON_EXEC" -m pip install -r requirements.txt
    
    echo "📦 Installing Frontend dependencies..."
    (cd "$PROJECT_ROOT/frontend" && npm install)
    
    if [ "$1" == "--install-only" ]; then
        echo "✅ Dependencies installed successfully."
        exit 0
    fi
fi

# Free up ports 8000 and 5173 if currently occupied
if command -v lsof &>/dev/null; then
    lsof -ti:8000,5173 2>/dev/null | xargs kill -9 2>/dev/null || true
fi

echo "✨ Launching Backend API & React Command Center..."
echo "🌐 Frontend Dashboard: http://localhost:5173"
echo "🌐 Backend API & Docs: http://localhost:8000/docs"
echo "========================================================"

export OPENCV_AVFOUNDATION_SKIP_AUTH=0
exec "$PYTHON_EXEC" src/orchestrate.py --mode dashboard "$@"
