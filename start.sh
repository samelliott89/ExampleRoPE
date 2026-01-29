#!/bin/bash
# Run this on vast.ai instance to start training

set -e  # exit on error

# Detect number of GPUs
NUM_GPUS=$(nvidia-smi -L | wc -l)
echo "Detected $NUM_GPUS GPUs"

# Install dependencies
echo "Installing dependencies..."
pip install -r requirements.txt

# Load .env if exists
if [ -f .env ]; then
    export $(cat .env | grep -v '^#' | xargs)
    echo "Loaded .env"
fi

# Start training
echo "Starting training..."
if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Using DDP with $NUM_GPUS GPUs"
    torchrun --nproc_per_node=$NUM_GPUS model.py
else
    echo "Single GPU training"
    python model.py
fi
