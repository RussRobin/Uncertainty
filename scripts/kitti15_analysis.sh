#!/usr/bin/env bash
set -e

python post_process/generate_statistic.py\
    --logdir ...\
    --epochs 499 \
    --maxdisp 192 \
    --inliers 5 \
    --mask hard \
    --dataset kitti\
    --model gwcnet-conor\
    --uncert