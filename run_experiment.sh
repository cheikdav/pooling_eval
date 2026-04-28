#!/bin/bash
# Run a complete experiment pipeline via Snakemake.
#
# Usage:
#   ./run_experiment.sh [CONFIG_PATH] [SNAKEMAKE_ARGS...]
#
# Default config: configs/test_mini/config.yaml
# Any extra arguments are forwarded to snakemake (e.g. -j 8, --dry-run).
set -euo pipefail

CONFIG="${1:-configs/test_mini/config.yaml}"
shift || true

echo "============================================"
echo "Running pipeline: $CONFIG"
echo "============================================"
uv run snakemake \
    --cores "${SNAKEMAKE_CORES:-4}" \
    --config experiment_config="$CONFIG" \
    "$@"

echo ""
echo "============================================"
echo "Pipeline complete."
echo "============================================"
