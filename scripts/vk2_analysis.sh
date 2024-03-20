#!/usr/bin/env bash
set -e

python post_process/generate_statistic.py\
    --logdir ... \
    --epochs 99 \
    --maxdisp 64 \
    --inliers 5 \
    --mask hard \
    --dataset vkitti\
    --model gwcnet-gcs\
    --uncert