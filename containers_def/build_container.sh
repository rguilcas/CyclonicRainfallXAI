#!/bin/bash
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --time=06:00:00
#SBATCH --account=nn12107k
#SBATCH --job-name=e2s_build_clean
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err
#SBATCH --mem-per-gpu=128G

mkdir -p logs
apptainer build --ignore-fakeroot-command /cluster/projects/nn12107k/robin/apptainer/earth2studio_v2.sif earth2studio.def