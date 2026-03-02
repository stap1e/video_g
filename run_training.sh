#!/bin/bash

# Set GPU ID (default: 2)
GPU_ID=${1:-2}

# Set CUDA visible devices
export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "Using GPU ID: $GPU_ID"
echo "Starting training..."

# Run training
python main.py

echo "Training completed."