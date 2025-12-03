#!/bin/bash
# Quick test of TracIn implementation on small subset
# Run this on login node to test before full-scale run

# Activate environment
source venv/bin/activate

# Test with tiny subset (should complete in <2 minutes)
echo "Testing TracIn on small subset..."
python TracIn/efficient_tracin.py \
  --train_subset 50 \
  --test_subset 10 \
  --top_k 20 \
  --batch_size 16 \
  --output_dir TracIn/test_results

echo "✓ Test complete! Check TracIn/test_results/"
