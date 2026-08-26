#!/bin/bash
# VERS Environment Restoration Script
# This script restores the VERS environment to its exact "perfect" working state.

echo "🚨 Starting VERS Environment Restoration..."

# 1. Kill existing processes
echo "--- Clearing port locks ---"
lsof -i :8501 -t | xargs kill -9 2>/dev/null
lsof -i :8000 -t | xargs kill -9 2>/dev/null

# 2. Rebuild Virtual Environment
echo "--- Rebuilding .venv ---"
rm -rf .venv
python3 -m venv .venv

# 3. Install Strict Dependencies
echo "--- Installing stabilized dependencies ---"
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install "numpy<2.0.0" "opencv-python<4.9" "opencv-contrib-python<4.9" "mediapipe==0.10.14" "protobuf~=4.25.3" --no-compile
./.venv/bin/python -m pip install -r requirements.txt --no-compile

# 4. Permissions Helper
chmod +x fix_camera.sh orchestrate.py

echo "✅ Environment Restored Successfully!"
echo "To run the dashboard: ./.venv/bin/python src/orchestrate.py --mode dashboard"
