#!/bin/bash
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --time=12:00:00
#SBATCH --account=nn12107k
#SBATCH --job-name=xai-aifs
#SBATCH --mem-per-gpu=128G
#SBATCH --exclude=gpu-1-111
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err

set -euo pipefail
mkdir -p logs

CONTAINER="/cluster/projects/nn12107k/robin/apptainer/earth2studio_v2.sif"
BINDDIRS="/cluster/home/rguilcas,/cluster/projects/nn12107k/"
SCRIPT="/cluster/home/rguilcas/code/CyclonicRainfall/CyclonicRainfallXAI/python_scripts_v3/xai_aifs.py"

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
            --region-name "California heatwave above38 2019-06-11T00"\
            --target-vars 2t \
            --target-time-start 2019-06-11T00:00 \
            --target-time-end 2019-06-11T18:00 \
            --lead-times 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18\
            --time-operation sum \
        #     --overwrite \
            --latlon

            #--overwrite