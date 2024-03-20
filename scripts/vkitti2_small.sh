#!/usr/bin/env bash
set -e
VK2PATH="..."
SAVEPATH="..."


python main.py --dataset vkitti2 \
   --training \
   --datapath $VK2PATH --trainlist ./filenames/vkitti2_train_small.txt --testlist ./filenames/vkitti2_test.txt \
   --maxdisp 64 \
   --ord_num 63 \
   --alpha 0.0 \
   --beta 63.0 \
   --gamma 0.0 \
   --epochs 100 --lrepochs " 7,20,30,40,45:5" \
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
   --logdir $SAVEPATH/conor/ \
   --test_batch_size 2\
   --device_id 0 1\
   --devices '0,1'\
   --discretization 'UD'\