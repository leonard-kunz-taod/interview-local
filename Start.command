#!/bin/bash
set -eu
cd "$(dirname "$0")"
if [ -d .cache/models ]; then export HF_HUB_CACHE="$PWD/.cache/models"; fi
if [ -d .cache/nltk_data ]; then export NLTK_DATA="$PWD/.cache/nltk_data"; fi
if [ ! -x .venv/bin/python ]; then
  echo 'Run Install.command first.'
else
  .venv/bin/python -m interview_local "$@" || true
fi
read -r -p 'Press Enter to close …' ignored
