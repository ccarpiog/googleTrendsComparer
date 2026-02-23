#!/usr/bin/env bash
# Launch script for the Google Trends Explorer Streamlit app.
# Creates a virtual environment and installs dependencies if needed,
# then starts the Streamlit server.

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
REQ_FILE="$SCRIPT_DIR/requirements.txt"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

# Install/update dependencies if requirements.txt is newer than the venv marker
MARKER="$VENV_DIR/.deps_installed"
if [ ! -f "$MARKER" ] || [ "$REQ_FILE" -nt "$MARKER" ]; then
    echo "Installing dependencies..."
    pip install -q -r "$REQ_FILE"
    touch "$MARKER"
fi

# Launch the app
echo "Starting Google Trends Explorer..."
streamlit run "$SCRIPT_DIR/app.py" "$@"
