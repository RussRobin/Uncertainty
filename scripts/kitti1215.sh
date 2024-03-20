#!/usr/bin/env bash
set -x
DATAPATH="..."
SAVEPATH="..."

python main.py --dataset kitti \
    --datapath $DATAPATH --trainlist ./filenames/kitti1215_train.txt --testlist ./filenames/kitti12_val.txt \
    --epochs 500 --lrepochs "184,300:5" \
    --save_freq 100\
    --maxdisp 64 \
    --ord_num 63 \
    --alpha 0.0 \
    --beta 63.0 \
    --gamma 0.0 \
    --model gwcnet-conor \
    --batch_size 4 \
    --training \
    --loss_type 'ConOR' \
    --save_test \
    --bin_scale 'log' \
    --n_bins 11 \
    --inliers 5 \
    --logdir $SAVEPATH/checkpoints/kitti1215/conor \
    --test_batch_size 2 \
    --device_id 0\
    --devices '0' \