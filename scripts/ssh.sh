#!/bin/bash
# SSH into vast.ai instance

set -e

# Project root (one level up from scripts/)
PROJECT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Load config
if [ -f "$PROJECT_DIR/vast.config" ]; then
    source "$PROJECT_DIR/vast.config"
else
    echo "Error: vast.config not found. Copy vast.config.example and edit it."
    exit 1
fi

echo "Connecting to $VAST_HOST..."
ssh -p "$VAST_PORT" "$VAST_HOST"
