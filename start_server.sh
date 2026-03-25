#!/bin/bash
# Start the Bili_Notes Flask server
# Usage: ./start_server.sh

# Change to the directory where this script lives (project root)
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Dependency installation
# ---------------------------------------------------------------------------
# Detect how to call pip: prefer `uv pip` if uv is available and a .venv
# exists (uv-managed project), otherwise fall back to plain pip/pip3.
if command -v uv >/dev/null 2>&1 && [ -d ".venv" ]; then
    PIP_CMD="uv pip install"
    # Activate the uv-managed virtual environment
    source .venv/bin/activate
elif command -v uv >/dev/null 2>&1 ; then
    uv venv
    source .venv/bin/activate
    PIP_CMD="uv pip install"
else command -v python3 >/dev/null 2>&1 ; then
    PIP_CMD="python3 -m pip install"
else
    echo "No available python3 environment."
    exit 1
fi

if [ -f "requirements.txt" ]; then
    echo "Setup Python dependencies..."
    $PIP_CMD -r requirements.txt --quiet
fi

# ---------------------------------------------------------------------------
# Create data directories if they don't exist
# ---------------------------------------------------------------------------
mkdir -p data/temp data/userdata

# ---------------------------------------------------------------------------
# Start the Flask server
# ---------------------------------------------------------------------------
python app.py
