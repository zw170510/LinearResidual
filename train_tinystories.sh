#!/bin/bash
# Quick start script for TinyStories training

echo "=== TinyStories Dataset Quick Start ==="
echo ""

# Check if data is prepared
if [ ! -f "data/tinystories/train.bin" ]; then
    echo "Step 1: Preparing TinyStories dataset..."
    echo "This will download and tokenize the dataset (one-time operation)"
    echo ""
    cd data/tinystories
    python prepare.py
    cd ../..
    echo "✓ Dataset prepared!"
    echo ""
else
    echo "✓ Dataset already prepared"
    echo ""
fi

echo "Step 2: Starting training..."
echo "Using configuration: config/train_tinystories.py"
echo ""
echo "To customize training, you can add arguments:"
echo "  --batch_size=16        : Adjust batch size"
echo "  --max_iters=200000     : Train longer"
echo "  --learning_rate=8e-4   : Change learning rate"
echo ""

python train.py config/train_tinystories.py "$@"
