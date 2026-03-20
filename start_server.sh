#!/bin/bash
# Start the a2doc Flask server
# Usage: ./start_server.sh

# Change to the directory where this script lives (project root)
cd "$(dirname "$0")"

# Activate virtual environment
source .venv/bin/activate

# Create data directories if they don't exist
mkdir -p data/temp data/userdata

# Start the Flask server
python app.py
