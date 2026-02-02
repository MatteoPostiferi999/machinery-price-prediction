#!/bin/bash
# setup_env.sh - Project-local version
# ─────────────────────────────────────────────────────────────────────

set -e

# Usiamo .venv dentro la cartella del progetto
VENV_NAME=".venv"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_PATH="$SCRIPT_DIR/$VENV_NAME"
REQUIREMENTS="$SCRIPT_DIR/requirements.txt"

# ── 1. Creazione ────────────────────────────────────────────────────
if [ -d "$VENV_PATH" ]; then
    echo "✓ Environment already exists in $VENV_PATH"
else
    echo "→ Creating local venv in $VENV_PATH ..."
    python3 -m venv "$VENV_PATH"
fi

# ── 2. Attivazione ──────────────────────────────────────────────────
# Il comando 'source' è fondamentale per agire sulla shell corrente
source "$VENV_PATH/bin/activate"

# ── 3. Aggiornamento e Installazione ────────────────────────────────
echo "→ Updating pip and installing dependencies..."
pip install --upgrade pip --quiet

if [ -f "$REQUIREMENTS" ]; then
    pip install -r "$REQUIREMENTS"
    echo "✓ Dependencies installed."
else
    echo "⚠ requirements.txt not found, skipping install."
fi

echo ""
echo "═══════════════════════════════════════════════"
echo "  SETUP COMPLETE"
echo "  The interpreter is now at: $VENV_PATH/bin/python"
echo "  VS Code should auto-detect it now."
echo "═══════════════════════════════════════════════"