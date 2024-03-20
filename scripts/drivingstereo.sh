#!/usr/bin/env bash
set -x
DATAPATH="..."
SAVEPATH="..."

python main.py --dataset drivingstereo \
    --datapath $DATAPATH --trainlist ./filenames/drivingstereo_train.txt --testlist ./filenames/drivingstereo_test_half_size.txt \
    --epochs 10 --lrepochs "5:5" \
    --save_freq 2\
    --maxdisp 64 \
    --ord_num 63 \
    --alpha 0.0 \
    --beta 63.0 \
    --gamma 0.0 \
    --model gwcnet-conor \
    --batch_size 8 \
    --training \
    --loss_type 'ConOR' \
    --save_test \
    --bin_scale 'log' \
    --n_bins 11 \
    --inliers 5 \
    --logdir $SAVEPATH/checkpoints/drivingstereo/conor \
    --test_batch_size 2 \
    --device_id 0 1\
    --devices '0,1'\
