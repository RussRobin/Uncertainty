#!/usr/bin/env bash
set -e

python post_process/generate_statistic.py\
    --logdir ...\
    --epochs 4 \
    --maxdisp 64 \
    --inliers 5 \
    --mask hard \
    --dataset sceneflow\
    --model gwcnet-gcs\
    --uncert