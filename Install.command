#!/bin/bash
set -eu
cd "$(dirname "$0")"
trap 'code=$?; if [ "$code" -ne 0 ]; then echo "Installation failed. Check the message above."; read -r -p "Press Enter to exit …" ignored; fi' EXIT
echo 'Interview Local: isolated installation (internet required).'
if ! command -v python3 >/dev/null 2>&1; then
  echo 'Install Python 3 from https://www.python.org/downloads/ first.'
  read -r -p 'Press Enter to exit …' ignored
  exit 1
fi
python3 -m venv .installer
.installer/bin/python -m pip install --disable-pip-version-check 'uv>=0.7.8,<1'
.installer/bin/uv sync --frozen --python 3.12
if [ -d /Applications/Ollama.app ]; then
  open -a Ollama
  sleep 2
  .venv/bin/python -m interview_local --prepare-llm || echo 'Open Ollama, then run: bash Start.command --prepare-llm'
else
  echo 'For readable drafts, install and open Ollama: https://ollama.com/download'
  echo 'Then run: bash Start.command --prepare-llm'
fi
printf '\nInstallation complete. Open Start.command next.\n' 
read -r -p 'Press Enter to exit …' ignored
