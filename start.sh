#!/usr/bin/env bash
# Start script for Render or Docker: migrate, create the admin user once, run gunicorn.
set -euo pipefail
python manage.py migrate --noinput
if [[ -n "${DJANGO_SUPERUSER_USERNAME:-}" && -n "${DJANGO_SUPERUSER_PASSWORD:-}" ]]; then
  python manage.py createsuperuser --noinput --email "${DJANGO_SUPERUSER_EMAIL:-admin@example.com}" 2>/dev/null || true
fi
exec gunicorn config.wsgi:application --bind "0.0.0.0:${PORT:-8000}" --workers "${WEB_CONCURRENCY:-2}" --timeout 60
