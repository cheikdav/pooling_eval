#!/bin/bash
# Simple script to run complete experiment pipeline

CONFIG="configs/example_config.yaml"

echo "============================================"
echo "Step 1: Training Policy"
echo "============================================"
uv run -m src.train_policy --config "$CONFIG"

echo ""
echo "============================================"
echo "Step 2: Generating Data"
echo "============================================"
uv run -m src.generate_data --config "$CONFIG"

echo ""
echo "============================================"
echo "Step 3: Training Estimators"
echo "============================================"
uv run -m src.run_all_estimators --config "$CONFIG" --mode sequential

echo ""
echo "============================================"
echo "Experiment Complete!"
echo "============================================"
echo "To evaluate results, run:"
echo "  python -m src.evaluate --config $CONFIG"
