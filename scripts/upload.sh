#!/bin/bash
# Upload project to vast.ai instance

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

echo "Uploading $PROJECT_DIR to $VAST_HOST:$REMOTE_DIR"

rsync -avz --progress \
    -e "ssh -p $VAST_PORT" \
    --exclude '.venv/' \
    --exclude 'venv/' \
    --exclude '__pycache__/' \
    --exclude '.git/' \
    --exclude 'data/' \
    --exclude 'checkpoints/' \
    --exclude 'vast_checkpoints/' \
    --exclude 'wandb/' \
    --exclude '*.pyc' \
    "$PROJECT_DIR/" \
    "$VAST_HOST:$REMOTE_DIR/"

echo ""
echo "Done. Now run:"
echo "  ssh -p $VAST_PORT $VAST_HOST"
echo "  cd $REMOTE_DIR && ./start.sh"
