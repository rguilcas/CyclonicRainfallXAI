#!/bin/bash
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --time=4:00:00
#SBATCH --account=nn12107k
#SBATCH --job-name=cali-heat-predictions
#SBATCH --mem-per-gpu=128G
#SBATCH --exclude=gpu-1-111
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err

set -euo pipefail
mkdir -p logs

CONTAINER="/cluster/projects/nn12107k/robin/apptainer/earth2studio_v2.sif"
BINDDIRS="/cluster/home/rguilcas,/cluster/projects/nn12107k/"
SCRIPT="/cluster/home/rguilcas/code/AIFS-XAI/scripts/predict_region.py"


run_day() {
    local day="$1"    # format: YYYY-MM-DD

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
        python "${SCRIPT}" \
            --region-name "California Central Valley" \
            --region-geojson-path /cluster/home/rguilcas/code/AIFS-XAI/aux/california_shapes.geojson \
            --target-vars 2t \
            --target-time-start "${day}T00:00" \
            --target-time-end "${day}T18:00" \
            --min-lead 1 \
            --max-lead 24 \
            --batch-size 4 \
            --time-operation max
        #     --overwrite
}

# Or loop over a range of days
days=(
    1940-08-11
    1942-06-30
    1942-07-23
    1943-07-27
    1944-08-27
    1945-07-26
    1946-08-02
    1950-06-30
    1950-08-17
    1950-09-01
    1954-06-22
    1955-09-02
    1960-06-02
    1960-07-01
    1961-06-15
    1961-07-10
    1964-06-24
    1970-07-03
    1971-08-10
    1972-06-30
    1972-07-13
    1975-07-24
    1977-07-31
    1978-08-06
    1980-07-25
    1988-07-18
    1990-08-07
    1991-07-03
    1993-08-01
    1998-08-04
    2006-07-21
    2007-07-05
    2008-07-09
    2009-06-28
    2013-06-29
    2014-06-08
    2014-07-26
    2016-07-25
    2017-06-18
    2017-08-27
    2017-09-01
    2018-07-25
    2019-06-11
    2019-08-15
    2020-08-14
    2020-09-06
    2021-06-18
    2021-07-09
    2021-07-29
    2022-06-25
    2022-07-17
    2022-08-03
    2022-08-31
    2023-07-01
    2023-07-16
    2023-07-21
    2023-08-15
    2024-07-03
    2024-07-11
    2024-07-23
    2025-08-09
)

for day in "${days[@]}"; do
    run_day "$day"
done
