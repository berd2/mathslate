#!/usr/bin/env bash
# One-shot setup for people new to Python: creates an isolated environment,
# installs MathSlate's complete beginner setup (Jupyter Lab, interactive
# controls, Gemini assistant and secure key storage), writes a starter
# notebook with the import already in it, and opens it.
set -euo pipefail

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 1
fi

# Its own name, not the conventional `.venv`: this script is run from a clone
# of the repository, where `.venv` is very likely the development environment
# (`pip install -e ".[dev,jupyter,marimo]"`) and `uv venv` would replace it.
ENV_DIR=".venv-mathslate"
PYTHON="$ENV_DIR/bin/python"
[ -x "$PYTHON" ] || PYTHON="$ENV_DIR/Scripts/python.exe"   # Git Bash on Windows

if [ ! -d "$ENV_DIR" ]; then
    uv venv "$ENV_DIR"
    PYTHON="$ENV_DIR/bin/python"
    [ -x "$PYTHON" ] || PYTHON="$ENV_DIR/Scripts/python.exe"
fi
echo "Installing MathSlate, Jupyter Lab, interactive controls, Gemini AI, and secure key storage..."
# Install this checkout rather than an older PyPI release: the script lives in
# the repository and is intended to make that exact version ready to use.
uv pip install --python "$PYTHON" ".[starter]"

NOTEBOOK="mathslate_quickstart.ipynb"
if [ ! -f "$NOTEBOOK" ]; then
    cat > "$NOTEBOOK" <<'EOF'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# MathSlate quickstart\n",
    "\n",
    "Run the next cell once, then try your own expression in a new cell — e.g. `plot(tan(x))` or `analyze(1/x)`."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from mathslate import *"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "plot(sin(x)/x)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "from mathslate.ai import assistant\n",
    "assistant()  # paste a Gemini API key in the panel"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "name": "python",
   "version": "3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
EOF
fi

# The environment's own interpreter, not `uv run`: run from a clone, `uv run`
# would find the repository's pyproject.toml and sync a *project* environment
# instead, which does not have the `jupyter` extra in it.
"$PYTHON" -m jupyter lab "$NOTEBOOK"
