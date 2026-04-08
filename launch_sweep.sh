#!/bin/bash
# Helper script to launch W&B sweeps for hyperparameter tuning

set -e

# Check if method argument is provided
if [ -z "$1" ]; then
    echo "Usage: ./launch_sweep.sh <method> [num_agents]"
    echo "Methods: monte_carlo, td"
    echo "Example: ./launch_sweep.sh monte_carlo 4"
    exit 1
fi

METHOD=$1
NUM_AGENTS=${2:-1}  # Default to 1 agent if not specified

SWEEP_CONFIG="configs/sweep_${METHOD}.yaml"

# Check if sweep config exists
if [ ! -f "$SWEEP_CONFIG" ]; then
    echo "Error: Sweep config not found: $SWEEP_CONFIG"
    exit 1
fi

echo "Creating sweep for ${METHOD}..."
SWEEP_ID=$(wandb sweep "$SWEEP_CONFIG" 2>&1 | grep -oE 'wandb agent [^ ]+' | awk '{print $3}')

if [ -z "$SWEEP_ID" ]; then
    echo "Error: Failed to create sweep"
    exit 1
fi

echo "Sweep created: $SWEEP_ID"
echo "Starting $NUM_AGENTS agent(s)..."

# Launch agents in parallel
for i in $(seq 1 $NUM_AGENTS); do
    echo "Starting agent $i..."
    wandb agent "$SWEEP_ID" &
done

# Wait for all agents to complete
wait

echo "All agents completed!"
