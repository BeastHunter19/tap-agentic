#!/bin/sh

set -e

# Get the secret file path from the environment variable
SECRET_FILE_PATH="${MAPS_API_KEY_FILE}"

if [ -z "${SECRET_FILE_PATH}" ]; then
  echo "Error: MAPS_API_KEY_FILE environment variable is not set."
  exit 1
fi

if [ -f "${SECRET_FILE_PATH}" ]; then
  # Read the entire file contents into MAPS_API_KEY, trimming any trailing newline
  GOOGLE_MAPS_API_KEY=$(cat "${SECRET_FILE_PATH}" | tr -d '\n')
else
  exit 1
fi

export GOOGLE_MAPS_API_KEY

exec mcp-google-map