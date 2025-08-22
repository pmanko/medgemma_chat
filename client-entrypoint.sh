#!/bin/sh
set -e

# Normalize line endings just in case
dos2unix 2>/dev/null || true

if [ -n "$API_BASE_URL" ]; then
  echo "Updating API base URL to: $API_BASE_URL"
  find /usr/share/nginx/html -type f -name "*.js" -exec sed -i "s|http://127.0.0.1:3000|$API_BASE_URL|g" {} \;
fi

exec nginx -g "daemon off;"


