#!/bin/bash
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --time=04:00:00
#SBATCH --account=nn12107k
#SBATCH --job-name=jupyter-job
#SBATCH --mem-per-gpu=128G
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err

set -euo pipefail
mkdir -p logs

CONTAINER="/cluster/projects/nn12107k/robin/apptainer/earth2studio_v2.sif"
BINDDIRS="/cluster/home/rguilcas,/cluster/projects/nn12107k/"

# Pick a port unlikely to collide with other users on the same node
PORT=$(shuf -i 20000-29999 -n 1)
NODE=$(hostname -s)

echo "=================================================="
echo "Jupyter will run on node: ${NODE}"
echo "Port: ${PORT}"
echo "To connect, run this on your LOCAL machine:"
echo "ssh -L ${PORT}:${NODE}:${PORT} <your_username>@olivia-login.sigma2.no -J <your_username>@olivia-login.sigma2.no"
echo "(adjust login hostname if different)"
echo "Then open the URL with the token shown below in your local browser."
echo "=================================================="

apptainer run --nv     \
                --bind "${BINDDIRS}"     \
                --env EARTH2STUDIO_CACHE=/cluster/projects/nn12107k/robin/earth2studio_cache     \
                --env HTTP_PROXY="${HTTP_PROXY:-}"     \
                --env HTTPS_PROXY="${HTTPS_PROXY:-}"    \
                 --env http_proxy="${http_proxy:-}"     \
                 --env https_proxy="${https_proxy:-}"   \
                 --env NO_PROXY="${NO_PROXY:-},s3.amazonaws.com,.s3.amazonaws.com"    \
                 --env no_proxy="${no_proxy:-},s3.amazonaws.com,.s3.amazonaws.com"     \
                 "${CONTAINER}"\
                 jupyter lab --no-browser --ip=0.0.0.0 --port="${PORT}"