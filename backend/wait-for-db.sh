#!/bin/sh
#
# backend/wait-for-db.sh
#
# This script waits until Postgres is accepting connections, using
# the POSTGRES_USER and POSTGRES_PASSWORD from the container's env.
# Once the DB is ready, it execs whatever command you pass to it.
#

set -e

# ─── Read settings from environment ───
DB_HOST="${DB_HOST:-db}"               # default “db” if not overridden
DB_PORT="${DB_PORT:-5432}"
DB_USER="${POSTGRES_USER:?Need to set POSTGRES_USER env var}"
DB_PASS="${POSTGRES_PASSWORD:?Need to set POSTGRES_PASSWORD env var}"

echo "→ Waiting for PostgreSQL at $DB_HOST:$DB_PORT (user=$DB_USER) …"

# Loop until psql can successfully connect
until PGPASSWORD="$DB_PASS" psql \
    -h "$DB_HOST" \
    -U "$DB_USER" \
    -p "$DB_PORT" \
    -c '\q' >/dev/null 2>&1; do

  echo "  ○ PostgreSQL is unavailable – sleeping 1s"
  sleep 1
done

echo "→ PostgreSQL is up — executing:"
echo "   $@"

# Finally, replace shell with the command we passed in
exec "$@"
