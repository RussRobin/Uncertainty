#!/usr/bin/env bash
set -x
DATAPATH="..."
SAVEPATH="..."


python main.py --dataset drivingstereo \
    --training \
    --datapath $DATAPATH --trainlist ./filenames/drivingstereo_train_small.txt --testlist ./filenames/drivingstereo_test_half_size.txt \
    --epochs 50 --lrepochs "7,20,30,40,45:5" \
    --save_freq 10\
    --maxdisp 64 \
    --ord_num 63 \
    --alpha 1.0 \
    --beta 64.0 \
    --gamma 0.0 \
    --model gwcnet-conor \
    --batch_size 12 \
    --loss_type 'ConOR' \
    --save_test \
    --bin_scale 'log' \
    --n_bins 11 \
    --inliers 5 \
    --logdir $SAVEPATH/checkpoints/drivingstereo-small/conor \
    --test_batch_size 2 \
    --device_id 0 1\
    --devices '0,1'\