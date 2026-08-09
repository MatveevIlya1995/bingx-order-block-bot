#!/usr/bin/env bash
# Loads .env and runs one bot cycle. Point cron/systemd at this script.
set -euo pipefail
cd "$(dirname "$0")"

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

python3 bot.py
