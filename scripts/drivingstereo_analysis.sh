#!/usr/bin/env bash
set -e

python post_process/generate_statistic.py --logdir ... \
    --epochs 49 \
    --maxdisp 64 \
    --inliers 5 \
    --mask 'soft' \
    --bin_scale 'log' \
    --n_bins 11 \
    --model gwcnet-conor\
    --uncert \
    --dataset 'drivingstereo'