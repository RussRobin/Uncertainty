from __future__ import print_function, division
import argparse
import os
import torch.backends.cudnn as cudnn
import time
from datasets import __datasets__
from models import __models__
from utils import *
from torch.utils.data import DataLoader
from skimage import io
from utils.visualization import *

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


from PIL import Image
import torchvision.transforms as transforms


cudnn.benchmark = True

parser = argparse.ArgumentParser(description='Group-wise Correlation Stereo Network (GwcNet)')
parser.add_argument('--model', default='gwcnet-gwc', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity')

parser.add_argument('--dataset', default='kitti',required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--testlist', required=True, help='testing list')
parser.add_argument('--loadckpt', required=True, help='load the weights from a specific checkpoint')

# OR args, please refer to "Deep Ordinal Regression Network for Monocular Depth Estimation" func. (1)
parser.add_argument('--discretization', type=str, default='UD', help='SID or UD')
parser.add_argument('--ord_num',required=True, type=int, default=192, help='in stereo matching, try to make ord_num = maxdisp, please refer to Deep Ordinal Regression func. (1)')
parser.add_argument('--alpha',required=True, type=float, default=1.0, help='please refer to Deep Ordinal Regression func. (1)')
parser.add_argument('--beta',required=True, type=float, default=192.0, help='please refer to Deep Ordinal Regression func. (1)')
parser.add_argument('--gamma',required=True, type=float, default=0.0, help=' alpha+gamma=1, please refer to Deep Ordinal Regression func. (1)')

# draw pdf and gt distribution
parser.add_argument('--drawpdf', action='store_true', help='draw pdf distribution')

#  for sceneflow
parser.add_argument('--zoom', type=float, default=1.0, help='scaler for zoom in/out the image')
parser.add_argument('--crop_w', type=int, default=0, help='random crop width')
parser.add_argument('--crop_h', type=int, default=0, help='random crop height')

# parse arguments
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

if ((args.dataset=='kitti12') or (args.dataset=='kitti15')):
    StereoDataset = __datasets__['kitti']
else:
    StereoDataset = __datasets__[args.dataset]
test_dataset = StereoDataset(args.datapath, args.testlist, False, args)
TestImgLoader = DataLoader(test_dataset, 1, shuffle=False, num_workers=8, drop_last=False)

if args.model in ['gwcnet-conor']: 
    cutpoints, t0s, ctpts, bin_values = find_cutpoints2tensor(args.discretization, args.ord_num, args.alpha, args.beta, args.gamma)
    # ctpts is t1s in ConOR
    conor_para = {'cutpoints': cutpoints, 't0s': t0s, 'ctpts':ctpts, 'bin_values':bin_values, 'ord_num':args.ord_num }
    model = __models__[args.model](args.maxdisp,conor_para['bin_values'])
else:
    model = __models__[args.model](args.maxdisp)
model = nn.DataParallel(model)
model.cuda()

print("loading model {}".format(args.loadckpt))
state_dict = torch.load(args.loadckpt)
model.load_state_dict(state_dict['model'])

def test():
    uncert_colormap = gen_error_colormap()

    os.makedirs('./predictions/disp_0', exist_ok=True)
    os.makedirs('./predictions/uncert', exist_ok=True)
    os.makedirs('./predictions/uncert_raw', exist_ok=True)
    os.makedirs('./predictions/gt', exist_ok=True)

    work_on_test_set = False
    if ('test' in str(args.testlist)) & ('vkitti2' not in str(args.testlist)) & ('drivingstereo' not in str(args.testlist)) & ('sceneflow' not in str(args.testlist)): 
        print('will not display errormap')
        work_on_test_set = True
    else:
        print('will display errormap')
        os.makedirs('./predictions/errormap', exist_ok=True)

    for batch_idx, sample in enumerate(TestImgLoader):        
        start_time = time.time()
        disp_est_tensor, uncert_est_tensor = test_sample(sample,args.model,args.drawpdf)

        uncert_est_np = tensor2numpy(uncert_est_tensor)
        top_pad_np = tensor2numpy(sample["top_pad"])
        right_pad_np = tensor2numpy(sample["right_pad"])
        left_filenames = sample["left_filename"]
        if not work_on_test_set:
            disp_gt = sample['disparity']

        disp_est_np = tensor2numpy(disp_est_tensor)

        print('Iter {}/{}, time = {:3f}'.format(batch_idx, len(TestImgLoader),
                                                time.time() - start_time))
        if not work_on_test_set:
            errormaps = [disp_error_image_func()(disp_est_tensor, disp_gt)]
        else:
            # this wouldn't make any change to the output
            errormaps = disp_est_np 
            disp_gt = disp_est_np
        for disp_est, uncert_est, top_pad, right_pad, fn, errormap_raw,disp_gt0 in zip(disp_est_np, uncert_est_np, top_pad_np, right_pad_np, left_filenames,errormaps,disp_gt):
            assert len(disp_est.shape) == 2
            disp_est = np.array(disp_est, dtype=np.float32)
            uncert_est = np.array(uncert_est, dtype=np.float32)
            disp_gt0 = np.array(disp_gt0.cpu(), dtype=np.float32)

            if (args.dataset == 'vkitti2'): # filter out sky in vk2
                mask = (disp_gt0 > 1) & (disp_gt0 < args.maxdisp)
            else:
                mask = (disp_gt0 > 0) & (disp_gt0 < args.maxdisp)
    

            uncert_est[np.logical_not(mask)] = 0.

            min_uncert = uncert_est[mask].min()
            max_uncert = uncert_est[mask].max()
            fn_uncert_raw = fn.split('/')[-1]
            fn_uncert_raw = os.path.join("predictions","uncert_raw", fn_uncert_raw.split('.')[0])
            np.save(fn_uncert_raw, uncert_est)

            if not work_on_test_set:
                errormap_raw = errormap_raw[0]
                errormap_raw  = np.array(errormap_raw)

                errormap_raw[:,np.logical_not(mask)]=0.
                errormap = errormap_raw.transpose(1,2,0)

                fn0 = os.path.join("predictions","errormap", fn.split('/')[-1])
                io.imsave(fn0, errormap)

            fn = os.path.join("predictions", fn.split('/')[-1])
            disp_est_uint = np.round(disp_est * 1).astype(np.uint8)
            disp_est_uint[np.logical_not(mask)] = 0
            io.imsave(fn, disp_est_uint)

            cols = uncert_colormap
            H, W = disp_gt0.shape
            uncert_image = np.zeros([H, W, 3], dtype=np.float32)
            if work_on_test_set:
                uncert_est[mask] = (uncert_est[mask] - min_uncert)*48/(max_uncert-min_uncert)
            else:
                uncert_est[mask] = (uncert_est[mask] - min_uncert)* 48/(max_uncert-min_uncert)
            for i in range(cols.shape[0]):
                uncert_image[np.logical_and(uncert_est/3. >= cols[i][0], uncert_est/3. < cols[i][1])] = cols[i, 2:]

            uncert_image[np.logical_not(mask)]=0.
            for i in range(cols.shape[0]):
                distance = 20
                uncert_image[ :10, i * distance:(i + 1) * distance, :] = cols[i, 2:]
            fn2 = os.path.join("predictions","uncert", fn.split('/')[-1])
            io.imsave(fn2, uncert_image)

            fn_gt = os.path.join("predictions","gt", fn.split('/')[-1])
            disp_gt0[np.logical_not(mask)] = 0.
            disp_gt0 = np.uint8(disp_gt0)
            io.imsave(fn_gt, disp_gt0)


# test one sample
@make_nograd_func
def test_sample(sample,model_type,draw_pdf=False):
    
    model.eval()
    output = model(sample['left'].cuda(), sample['right'].cuda())
    disp_ests = output['disp']
    uncert_ests = output['uncert']

    if model_type not in ['gwcnet-conor']:
        return disp_ests[-1], torch.exp(uncert_ests[-1])
    else: 
        return disp_ests[-1], uncert_ests[-1]

if __name__ == '__main__':
    test()
