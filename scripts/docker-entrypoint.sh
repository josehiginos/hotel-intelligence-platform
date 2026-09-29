#!/usr/bin/env bash
set -euo pipefail

cd /app

# Rebuild SQLite from committed CSVs when missing (hotel.db is gitignored).
if [[ ! -f data/hotel.db ]]; then
  python -c "from src.module_3_conversational.database.init_db import initialize_database; initialize_database()"
fi

exec "$@"
