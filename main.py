from __future__ import print_function, division
import argparse
import os
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import time
from tensorboardX import SummaryWriter
from datasets import __datasets__
from models import __models__, model_loss
from utils import *
from torch.utils.data import DataLoader
import gc


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

cudnn.benchmark = True

# model params
parser = argparse.ArgumentParser(description='Group-wise Correlation Stereo Network (GwcNet)')
parser.add_argument('--model', default='gwcnet-g', help='select a model structure', choices=__models__.keys())
parser.add_argument('--maxdisp', type=int, default=192, help='maximum disparity') # sclar down disp with H and W
parser.add_argument('--inliers', type=int, default=3, help='how many std to include for inliers')
parser.add_argument('--bin_scale', type=str, default='line', help='how to create the distribution, line or log')
parser.add_argument('--n_bins', type=int, default=11, help='how many bins to create the distribution')
parser.add_argument('--loss_type', type=str, required=True, help='define the componet of loss: ConOR+smooth_l1, ConOR, smooth_l1, KG or UC')
parser.add_argument('--mask', type=str, default='soft', help='type of mask assignment',choices=['soft','hard'])

# gpu
parser.add_argument("--devices", type=str, default="0,1",help="Comma-separated list of GPU IDs")

# dataset
parser.add_argument('--dataset', required=True, help='dataset name', choices=__datasets__.keys())
parser.add_argument('--zoom', type=float, default=1.0, help='scaler for zoom in/out the image')
parser.add_argument('--crop_w', type=int, default=0, help='random crop width')
parser.add_argument('--crop_h', type=int, default=0, help='random crop height')
parser.add_argument('--datapath', required=True, help='data path')
parser.add_argument('--trainlist', required=True, help='training list')
parser.add_argument('--testlist', required=True, help='testing list')

# training schedule
parser.add_argument('--training', action='store_true', help='turn to training mode if presents.')
parser.add_argument('--resume', action='store_true', help='continue training the model')
parser.add_argument('--loadckpt', help='load the weights from a specific checkpoint')
parser.add_argument('--seed', type=int, default=999, metavar='S', help='random seed (default: 1)')
parser.add_argument('--device_id', default=[0], type=int, nargs='+', help='gpu indices')
parser.add_argument('--batch_size', type=int, default=2, help='training batch size')
parser.add_argument('--test_batch_size', type=int, default=2, help='testing batch size')
parser.add_argument('--epochs', type=int, required=True, help='number of epochs to train')
parser.add_argument('--lr', type=float, default=0.0001, help='base learning rate')
parser.add_argument('--lrepochs', type=str, required=True, help='the epochs to decay lr: the downscale rate')

# save outputs
parser.add_argument('--logdir', required=True, help='the directory to save logs and checkpoints')
parser.add_argument('--summary_freq', type=int, default=10, help='the frequency of saving summary')
parser.add_argument('--save_freq', type=int, default=1, help='the frequency of saving checkpoint')
parser.add_argument('--save_test', action='store_true', help='save test outputs if presents.')

# OR args, please refer to "Deep Ordinal Regression Network for Monocular Depth Estimation" func. (1)
parser.add_argument('--discretization', type=str, default='UD', help='SID or UD')
parser.add_argument('--ord_num',required=True, type=int, default=192, help='in stereo matching, try to make ord_num = maxdisp, please refer to Deep Ordinal Regression func. (1)')
parser.add_argument('--alpha',required=True, type=float, default=1.0, help='please refer to Deep Ordinal Regression func. (1)')
parser.add_argument('--beta',required=True, type=float, default=192.0, help='please refer to Deep Ordinal Regression func. (1)')
parser.add_argument('--gamma',required=True, type=float, default=0.0, help=' alpha+gamma=1, please refer to Deep Ordinal Regression func. (1)')

# finetune
parser.add_argument('--finetune_different_models',required=False, type=bool, default=False, help='finetune ConOR based on a trained SEDNet')

# train on pixels with small data uncert
parser.add_argument('--small_uncert_train', action='store_true', help='only train on small data uncert')
parser.add_argument('--small_uncert_start', type=int, default=50, help='after which epoch to start training on small data uncert')


# parse arguments, set seeds
args = parser.parse_args()
torch.cuda.empty_cache()
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
os.makedirs(args.logdir, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

print("creating new summary file")
logger = SummaryWriter(args.logdir)

args.maxdisp = int(args.maxdisp * args.zoom)
StereoDataset = __datasets__[args.dataset]
train_dataset = StereoDataset(args.datapath, args.trainlist, True, args)
test_dataset = StereoDataset(args.datapath, args.testlist, False, args)
TrainImgLoader = DataLoader(train_dataset, args.batch_size, shuffle=True, num_workers=8, drop_last=True)
TestImgLoader = DataLoader(test_dataset, args.test_batch_size, shuffle=False, num_workers=4, drop_last=False)

if args.model in ['gwcnet-conor']: 
    cutpoints, t0s, ctpts, bin_values = find_cutpoints2tensor(args.discretization, args.ord_num, args.alpha, args.beta, args.gamma)
    conor_para = {'cutpoints': cutpoints, 't0s': t0s, 'ctpts':ctpts, 'bin_values':bin_values, 'ord_num':args.ord_num }
    model = __models__[args.model](args.maxdisp,conor_para['t0s'])

else:
    model = __models__[args.model](args.maxdisp)
    conor_para = None
model.cuda()

model = nn.DataParallel(model,device_ids=args.device_id)
optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999))

start_epoch = 0
if args.resume:
    assert args.finetune_different_models==False
    if args.loadckpt:
        print("loading model {}".format(args.loadckpt))
        state_dict = torch.load(args.loadckpt)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
    else:
        all_saved_ckpts = [fn for fn in os.listdir(args.logdir) if fn.endswith(".ckpt")]
        all_saved_ckpts = sorted(all_saved_ckpts, key=lambda x: int(x.split('_')[-1].split('.')[0]))
        loadckpt = os.path.join(args.logdir, all_saved_ckpts[-1])
        print("loading the lastest model in logdir: {}".format(loadckpt))
        state_dict = torch.load(loadckpt)
        model.load_state_dict(state_dict['model'])
        optimizer.load_state_dict(state_dict['optimizer'])
        start_epoch = state_dict['epoch'] + 1
elif args.loadckpt:
    print("loading model {}".format(args.loadckpt))
    print('do not load optimizer or start_epoch')
    state_dict = torch.load(args.loadckpt)
    if args.finetune_different_models:
        model.load_state_dict(state_dict['model'], strict=False)
    else:
        model.load_state_dict(state_dict['model'])
print("start at epoch {}".format(start_epoch))


def train():
    for epoch_idx in range(start_epoch, args.epochs):
        adjust_learning_rate(optimizer, epoch_idx, args.lr, args.lrepochs)
        # training
        if args.training:
            save_outputs = {"disp_est": [], "disp_gt": [], "uncert_est": [], "cost_conf": [], "pred_conf": []}
            for batch_idx, sample in enumerate(TrainImgLoader):
                global_step = len(TrainImgLoader) * epoch_idx + batch_idx
                start_time = time.time()
                do_summary = global_step % args.summary_freq == 0
                
                train_on_small_uncert = False                    
                if args.small_uncert_train & epoch_idx > args.small_uncert_start:
                    train_on_small_uncert = True                    
                losses, scalar_outputs, image_outputs = train_sample(sample, compute_metrics=do_summary,small_uncert=train_on_small_uncert)
                
                if do_summary:
                    save_scalars(logger, 'train', scalar_outputs, global_step)
                    save_images(logger, 'train', image_outputs, global_step)
                del scalar_outputs, image_outputs
                
                if (batch_idx%100==0) or (batch_idx<2):
                    print('Epoch {}/{}, Iter {}/{}, train loss = {:.3f}, time = {:.3f}'.format(epoch_idx, args.epochs,
                                                                                           batch_idx,
                                                                                           len(TrainImgLoader), losses["loss"], time.time() - start_time))
            # saving checkpoints
            if (epoch_idx + 1) % args.save_freq == 0:
                checkpoint_data = {'epoch': epoch_idx, 'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
                torch.save(checkpoint_data, "{}/checkpoint_{:0>6}.ckpt".format(args.logdir, epoch_idx))
            gc.collect()

        # testing
        avg_test_scalars = AverageMeterDict()
        save_outputs = {"disp_est":[], "disp_gt": [], "uncert_est":[], "cost_conf":[], "pred_conf":[]}
        for batch_idx, sample in enumerate(TestImgLoader):
            global_step = len(TestImgLoader) * epoch_idx + batch_idx
            start_time = time.time()
            do_summary = global_step % args.summary_freq == 0
            losses, scalar_outputs, image_outputs = test_sample(sample, compute_metrics=do_summary)
            if do_summary:
                save_scalars(logger, 'test', scalar_outputs, global_step)
                save_images(logger, 'test', image_outputs, global_step)
            avg_test_scalars.update(scalar_outputs)
            # save test result
            if args.save_test:
                if epoch_idx == (args.epochs - 1):
                    save_outputs["disp_est"].append(image_outputs["disp_est"][0].cpu())
                    save_outputs["disp_gt"].append(image_outputs["disp_gt"].cpu())
                    if args.model in ['gwcnet-gcs','gwcnet-conor']:
                        save_outputs["uncert_est"].append(image_outputs["uncert_est"][0].cpu())
            del scalar_outputs, image_outputs
            if (batch_idx%100==0) or (batch_idx<2):
                print('Epoch {}/{}, Iter {}/{}, test loss = {:.3f}, time = {:3f}'.format(epoch_idx, args.epochs,
                                                                             batch_idx,
                                                                                     len(TestImgLoader), losses["loss"],
                                                                                     time.time() - start_time))
        avg_test_scalars = avg_test_scalars.mean()
        save_scalars(logger, 'fulltest', avg_test_scalars, len(TrainImgLoader) * (epoch_idx + 1))
        if args.save_test:
            if epoch_idx == (args.epochs -1):
                torch.save(save_outputs, "{}/test_outputs_{:0>6}.pth".format(args.logdir, epoch_idx))
                save_outputs = {"disp_est":[], "disp_gt": [], "uncert_est":[], "cost_conf":[], "pred_conf":[]}

        print("avg_test_scalars", avg_test_scalars)
        gc.collect()


# train one sample
def train_sample(sample, compute_metrics=False,small_uncert=False):
    model.train()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    optimizer.zero_grad()

    output = model(imgL, imgR)

    disp_ests = output['disp']
    mask = (disp_gt < args.maxdisp) & (disp_gt > 0) 

    if small_uncert:
        uncert = output['uncert'][-1]
        mean_uncert = uncert[mask].mean()
        mask = (disp_gt < args.maxdisp) & (disp_gt > 0) & (uncert < mean_uncert) 

    losses = model_loss(output, disp_gt, mask, args,conor_para)

    scalar_outputs = {}
    for key in losses.keys():
        scalar_outputs[key] = losses[key]
    image_outputs = {"disp_est": disp_ests,
                     "disp_gt": disp_gt,
                     "imgL": imgL,
                     "imgR": imgR,
                    }

    if args.model in ['gwcnet-gcs']:
        uncert_ests = output['uncert']
        image_outputs["uncert_est"] = [torch.exp(uncert_ests[0])] 
  
    if compute_metrics:
        with torch.no_grad():
            image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]
            scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
            scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
            scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
            scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]
    losses["loss"].backward()
    optimizer.step()

    return tensor2float(losses), tensor2float(scalar_outputs), image_outputs


# test one sample
@make_nograd_func
def test_sample(sample, compute_metrics=True):
    model.eval()

    imgL, imgR, disp_gt = sample['left'], sample['right'], sample['disparity']
    imgL = imgL.cuda()
    imgR = imgR.cuda()
    disp_gt = disp_gt.cuda()

    output = model(imgL, imgR)
    disp_ests = output['disp']
    mask  = (disp_gt < args.maxdisp) & (disp_gt > 0)

    losses = model_loss(output, disp_gt, mask, args,conor_para)

    scalar_outputs = {}
    for key in losses.keys():
        scalar_outputs[key] = losses[key]
    image_outputs = {"disp_est": disp_ests,
                     "disp_gt": disp_gt,
                     "imgL": imgL,
                     "imgR": imgR,
                     }
    if args.model in ['gwcnet-gcs']:
        uncert_ests = output['uncert'][0]
        image_outputs["uncert_est"] = [torch.exp(uncert_ests)]
    elif args.model in ['gwcnet-conor']:
        uncert_ests = output['uncert']
        image_outputs["uncert_est"] = uncert_ests

    scalar_outputs["D1"] = [D1_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["EPE"] = [EPE_metric(disp_est, disp_gt, mask) for disp_est in disp_ests]
    scalar_outputs["Thres1"] = [Thres_metric(disp_est, disp_gt, mask, 1.0) for disp_est in disp_ests]
    scalar_outputs["Thres2"] = [Thres_metric(disp_est, disp_gt, mask, 2.0) for disp_est in disp_ests]
    scalar_outputs["Thres3"] = [Thres_metric(disp_est, disp_gt, mask, 3.0) for disp_est in disp_ests]

    if compute_metrics:
        image_outputs["errormap"] = [disp_error_image_func()(disp_est, disp_gt) for disp_est in disp_ests]
    
    return tensor2float(losses), tensor2float(scalar_outputs), image_outputs


if __name__ == '__main__':
    train()
