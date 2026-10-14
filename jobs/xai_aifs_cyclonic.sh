#!/bin/bash
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --time=10:00:00
#SBATCH --account=nn12107k
#SBATCH --job-name=cyclone-20250224_035_AR-363
#SBATCH --mem-per-gpu=128G
#SBATCH --exclude=gpu-1-111
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err

set -euo pipefail
mkdir -p logs

CONTAINER="/cluster/projects/nn12107k/robin/apptainer/earth2studio_v2.sif"
BINDDIRS="/cluster/home/rguilcas,/cluster/projects/nn12107k/"
SCRIPT="/cluster/home/rguilcas/code/AIFS-XAI/scripts/xai_aifs.py"

apptainer exec --nv \
            --bind "${BINDDIRS}" \
            --env EARTH2STUDIO_CACHE=/cluster/projects/nn12107k/robin/earth2studio_cache \
            --env HTTP_PROXY="${HTTP_PROXY:-}" \
            --env HTTPS_PROXY="${HTTPS_PROXY:-}" \
            --env http_proxy="${http_proxy:-}" \
            --env https_proxy="${https_proxy:-}" \
            --env NO_PROXY="${NO_PROXY:-},s3.amazonaws.com,.s3.amazonaws.com" \
            --env no_proxy="${no_proxy:-},s3.amazonaws.com,.s3.amazonaws.com" \
            --env LOGURU_LEVEL=WARNING \
    "${CONTAINER}" \
    python "${SCRIPT}"\
            --region-name "20250224_035_AR-363"\
            --region-geojson-path /cluster/home/rguilcas/code/AIFS-XAI/aux/xai_targets_AR_DJF-2024-2025_wgs84.geojson\
            --target-vars tp \
            --target-time-start 2025-02-24T00:00 \
            --target-time-end 2025-02-24T18:00 \
            --lead-times 1 \
            --ig-steps 20 \
            --time-operation sum \
            --overwrite
        #     --grad-scale 1000.0
        #     --overwrite
            