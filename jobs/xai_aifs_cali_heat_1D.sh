#!/bin/bash
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --time=6:00:00
#SBATCH --account=nn12107k
#SBATCH --job-name=cali-heat_above39
#SBATCH --mem-per-gpu=128G
#SBATCH --exclude=gpu-1-111
#SBATCH --output=logs/%x_%A.out
#SBATCH --error=logs/%x_%A.err

set -euo pipefail
mkdir -p logs

CONTAINER="/cluster/projects/nn12107k/robin/apptainer/earth2studio_v2.sif"
BINDDIRS="/cluster/home/rguilcas,/cluster/projects/nn12107k/"
SCRIPT="/cluster/home/rguilcas/code/AIFS-XAI/scripts/xai_aifs.py"


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
            --lead-times 32 31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 9 8 7 6 5 4 3 2 1 \
            --ig-steps 20 \
            --time-operation max \
            --grad-scale 1000.0
        #     --overwrite
}

# Single day
run_day 1970-07-03

# # Or loop over a range of days
# days=(
#     1940-08-11
#     1942-06-30
#     1942-07-23
#     1942-08-22
#     1943-07-27
#     1944-08-27
#     1945-06-19
#     1945-07-26
#     1945-08-21
#     1946-08-02
#     1950-06-30
#     1950-08-17
#     1954-06-22
#     1954-07-28
#     1960-06-02
#     1960-06-13
#     1960-07-19
#     1961-06-15
#     1961-07-10
#     1964-06-24
#     1966-06-14
#     1970-06-01
#     1970-07-03
#     1971-08-10
#     1972-06-30
#     1972-07-13
#     1973-06-07
#     1975-07-24
#     1977-06-06
#     1977-07-31
#     1978-08-06
#     1979-06-11
#     1979-07-31
#     1980-07-25
#     1985-06-11
#     1987-08-02
#     1988-07-18
#     1990-08-07
#     1991-07-03
#     1992-08-16
#     1993-08-01
#     1994-08-15
#     1996-06-03
#     1998-08-04
#     2006-07-21
#     2007-06-14
#     2007-07-05
#     2007-08-30
#     2008-07-09
#     2008-08-28
#     2009-06-28
#     2013-06-29
#     2014-06-08
#     2014-07-26
#     2016-07-25
#     2017-06-18
#     2017-08-27
#     2018-07-25
#     2018-08-18
#     2019-06-11
#     2019-08-15
#     2019-08-26
#     2020-08-14
#     2021-06-01
#     2021-06-18
#     2021-07-09
#     2021-07-29
#     2021-08-15
#     2021-08-28
#     2022-06-25
#     2022-07-17
#     2022-08-03
#     2022-08-16
#     2022-08-31
#     2023-07-01
#     2023-07-16
#     2023-07-21
#     2023-08-15
#     2024-07-03
#     2024-07-11
#     2024-07-23
#     2025-08-09
#     2025-08-21
#     2025-08-30
# )

# for day in "${days[@]}"; do
#     run_day "$day"
# done
