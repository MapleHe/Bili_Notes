#!/bin/bash
# Start the Bili_Notes Flask server
# Usage: ./start_server.sh

# Change to the directory where this script lives (project root)
cd "$(dirname "$0")"

# ---------------------------------------------------------------------------
# Detect environment
# ---------------------------------------------------------------------------
IS_TERMUX=false
if [ -n "$TERMUX_VERSION" ] || [ -d "/data/data/com.termux" ]; then
    IS_TERMUX=true
fi

# ---------------------------------------------------------------------------
# Dependency installation
# ---------------------------------------------------------------------------
if $IS_TERMUX; then
    echo "[Termux] Detected Termux environment."

    # Step 1: Install packages that must be compiled against Termux's sysroot
    # via pkg (pre-built binaries). pip cannot build these reliably on Android.
    echo "[Termux] Installing system packages via pkg (numpy, ninja)..."
    pkg install -y python-numpy ninja 2>/dev/null || {
        echo "[Termux] WARNING: pkg install failed or pkg is unavailable. Continuing anyway."
    }

    # Step 2: Install curl_cffi with --pre — stable releases have no ARM64 wheel.
    # Must run BEFORE the requirements.txt pass so pip does not later overwrite it
    # with a stable release that lacks an ARM64 wheel.
    echo "[Termux] Installing curl_cffi (pre-release, required for ARM64)..."
    pip install curl_cffi --pre --quiet || {
        echo "[Termux] WARNING: curl_cffi --pre failed. Falling back to plain requests."
    }

    # Step 3: Install the rest of requirements.txt.
    # pip skips packages that are already satisfied (numpy from pkg, curl_cffi from
    # step 2), so no special filtering is needed.
    if [ -f "requirements.txt" ]; then
        echo "[Termux] Checking remaining Python dependencies..."
        pip install -r requirements.txt --quiet || {
            echo "[Termux] WARNING: Some packages from requirements.txt failed to install."
        }
    fi

    PY_CMD="python3"

else
    # -------------------------------------------------------------------------
    # Non-Termux: standard uv / venv flow
    # -------------------------------------------------------------------------
    if command -v uv >/dev/null 2>&1 && [ -d ".venv" ]; then
        PIP_CMD="uv pip install"
        # shellcheck disable=SC1091
        source .venv/bin/activate
    elif [ -d ".venv" ]; then
        # shellcheck disable=SC1091
        source .venv/bin/activate
        PIP_CMD="pip install"
    elif command -v pip3 >/dev/null 2>&1; then
        PIP_CMD="pip3 install"
    else
        PIP_CMD="python3 -m pip install"
    fi

    if [ -f "requirements.txt" ]; then
        echo "Checking Python dependencies..."
        $PIP_CMD -r requirements.txt --quiet
    fi

    PY_CMD="python3"
    # If inside a venv, `python` is the venv interpreter; prefer it.
    command -v python >/dev/null 2>&1 && PY_CMD="python"
fi

# ---------------------------------------------------------------------------
# Create data directories if they don't exist
# ---------------------------------------------------------------------------
mkdir -p data/temp data/userdata

# ---------------------------------------------------------------------------
# Start the Flask server
# ---------------------------------------------------------------------------
$PY_CMD app.py
