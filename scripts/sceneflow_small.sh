#!/usr/bin/env bash
set -e
DATAPATH="..."
SAVEPATH="..."

python main.py --dataset sceneflow \
   --training \
   --datapath $DATAPATH --trainlist ./filenames/sceneflow_train_small.txt --testlist ./filenames/sceneflow_test.txt \
   --maxdisp 64 \
   --ord_num 63 \
   --alpha 1.0 \
   --beta 64.0 \
   --gamma 0.0 \
   --epochs 50 --lrepochs " 7,20,30,40,45:5" \
   --save_freq 10\
   --batch_size 6 \
   --lr 0.0001 \
   --loss_type 'ConOR' \
   --save_test \
   --mask 'soft' \
   --bin_scale 'log' \
   --n_bins 11 \
   --inliers 3 \
   --model gwcnet-conor \
   --logdir $SAVEPATH/checkpoints/sceneflow-small/conor \
   --test_batch_size 2\
   --device_id 0 1\
   --devices '0,1'\
   --discretization 'UD'\
   --zoom 1.0 \
   --crop_w 448 \
   --crop_h 284 \