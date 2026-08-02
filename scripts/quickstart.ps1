# One-shot setup for people new to Python: creates an isolated environment,
# installs mathslate with the Jupyter extra, writes a starter notebook with
# the import already in it, and opens it.
$ErrorActionPreference = "Stop"

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Error "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
}

# Its own name, not the conventional `.venv`: this script is run from a clone
# of the repository, where `.venv` is very likely the development environment
# (`pip install -e ".[dev,jupyter,marimo]"`) and `uv venv` would replace it.
$envDir = ".venv-mathslate"
$python = Join-Path $envDir "Scripts\python.exe"

if (-not (Test-Path $envDir)) {
    uv venv $envDir
}
uv pip install --python $python "mathslate[jupyter]"

$notebook = "mathslate_quickstart.ipynb"
if (-not (Test-Path $notebook)) {
    $content = @'
{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# MathSlate quickstart\n",
    "\n",
    "Run the next cell once, then try your own expression in a new cell -- e.g. `plot(tan(x))` or `analyze(1/x)`."
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
'@
    # Not Set-Content -Encoding utf8: on Windows PowerShell that writes a BOM,
    # and a notebook opening with one is a JSON parse error, not a notebook.
    [System.IO.File]::WriteAllText(
        (Join-Path (Get-Location).Path $notebook), $content
    )
}

# The environment's own interpreter, not `uv run`: run from a clone, `uv run`
# would find the repository's pyproject.toml and sync a *project* environment
# instead, which does not have the `jupyter` extra in it.
& $python -m jupyter lab $notebook
