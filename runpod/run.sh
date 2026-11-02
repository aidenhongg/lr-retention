#!/usr/bin/env bash
set -euo pipefail
cd /opt/mock-experiment
source /opt/pilot-env/bin/activate
export HF_HUB_DISABLE_PROGRESS_BARS=1
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MPLBACKEND=Agg
python -u -m src.experiments.sweep --action all --config configs/experiment.yaml --root . --artifacts /artifacts
