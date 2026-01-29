#!/bin/bash
# Download checkpoints from vast.ai instance

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

LOCAL_CHECKPOINTS="$PROJECT_DIR/vast_checkpoints"
mkdir -p "$LOCAL_CHECKPOINTS"

echo "Downloading checkpoints from $VAST_HOST:$REMOTE_DIR/checkpoints/"
echo "Saving to $LOCAL_CHECKPOINTS/"

rsync -avz --progress \
    -e "ssh -p $VAST_PORT" \
    "$VAST_HOST:$REMOTE_DIR/checkpoints/" \
    "$LOCAL_CHECKPOINTS/"

echo ""
echo "Done. Checkpoints saved to vast_checkpoints/"
ls -lh "$LOCAL_CHECKPOINTS/"
