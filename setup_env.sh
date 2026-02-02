#!/bin/bash
# setup_env.sh
# Creates (or reuses) a venv named "ML" in your home directory and installs dependencies.
# Usage: bash setup_env.sh
# ─────────────────────────────────────────────────────────────────────

set -e

VENV_NAME="ML"
VENV_PATH="$HOME/$VENV_NAME"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"   # folder where this script is located
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"

# ── 1. Create venv only if it doesn't exist ──────────────────────────
if [ -d "$VENV_PATH" ]; then
    echo "✓ Virtual environment '$VENV_NAME' already exists in $VENV_PATH"
else
    echo "→ Creating virtual environment '$VENV_NAME' in $VENV_PATH ..."
    python3 -m venv "$VENV_PATH"
    echo "✓ Environment created."
fi

# ── 2. Activate venv ───────────────────────────────────────────────
echo "→ Activating environment ..."
# shellcheck source=/dev/null
source "$VENV_PATH/bin/activate"
echo "✓ Environment activated. (Python: $(which python3))"

# ── 3. Upgrade pip ──────────────────────────────────────────────────
echo "→ Upgrading pip ..."
pip install --upgrade pip --quiet

# ── 4. Install dependencies ─────────────────────────────────────────
if [ -f "$REQUIREMENTS" ]; then
    echo "→ Installing dependencies from requirements.txt ..."
    pip install -r "$REQUIREMENTS"
    echo "✓ Dependencies installed."
else
    echo "⚠  requirements.txt not found in $SCRIPT_DIR — skipping installation."
fi

# ── 5. Summary ──────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════"
echo "  'ML' environment is ready."
echo "  To activate it in the future, run:"
echo "    source $VENV_PATH/bin/activate"
echo "  To deactivate it:"
echo "    deactivate"
echo "═══════════════════════════════════════════════"