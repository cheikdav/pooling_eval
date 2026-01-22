#!/bin/bash
set -e

# Run all estimator training jobs in parallel across GPUs
# Usage: ./run_parallel_estimators.sh <config_file> <methods> <n_batches> <overwrite> <timestamp>
#   methods: comma-separated list (e.g., "monte_carlo,dqn")
#   overwrite: "true" or "false"
#   timestamp: training session timestamp

if [ $# -lt 5 ]; then
    echo "Usage: $0 <config_file> <methods> <n_batches> <overwrite> <timestamp>"
    echo "Example: $0 configs/example.yaml monte_carlo,dqn 10 false 20240101_120000"
    exit 1
fi

CONFIG="$1"
METHODS="$2"
N_BATCHES="$3"
OVERWRITE="$4"
TIMESTAMP="$5"

if [ "$OVERWRITE" = "true" ]; then
    OVERWRITE_FLAG="--overwrite"
else
    OVERWRITE_FLAG="--no-overwrite"
fi

if [ ! -f "$CONFIG" ]; then
    echo "Error: Config file not found: $CONFIG"
    exit 1
fi

echo "Config: $CONFIG"
echo "Methods: $METHODS"
echo "Batches: 0 to $((N_BATCHES - 1))"
echo ""

# Detect GPUs
if command -v nvidia-smi &> /dev/null; then
    N_GPUS=$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l)
    echo "Found $N_GPUS GPUs"
else
    echo "No GPUs detected, running on CPU"
    N_GPUS=1
fi
echo ""

# Convert comma-separated methods to array
IFS=',' read -ra METHOD_ARRAY <<< "$METHODS"

# Run jobs distributed across GPUs
job_count=0
for method in "${METHOD_ARRAY[@]}"; do
    for batch_idx in $(seq 0 $((N_BATCHES - 1))); do
        if [ $N_GPUS -gt 1 ]; then
            gpu_idx=$((job_count % N_GPUS))
            echo "[GPU $gpu_idx] Starting: method=$method batch=$batch_idx"
            CUDA_VISIBLE_DEVICES=$gpu_idx python -m src.train_estimator \
                --config "$CONFIG" \
                --method "$method" \
                --batch-idx "$batch_idx" \
                $OVERWRITE_FLAG \
                --timestamp "$TIMESTAMP" &
        else
            echo "Starting: method=$method batch=$batch_idx"
            python -m src.train_estimator \
                --config "$CONFIG" \
                --method "$method" \
                --batch-idx "$batch_idx" \
                $OVERWRITE_FLAG \
                --timestamp "$TIMESTAMP" &
        fi

        job_count=$((job_count + 1))

        # Simple throttling: wait if we've launched N_GPUS jobs
        if [ $N_GPUS -gt 1 ] && [ $((job_count % N_GPUS)) -eq 0 ]; then
            wait -n || true
        fi
    done
done

# Wait for all remaining jobs
wait

echo ""
echo "All jobs completed!"
