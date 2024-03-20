import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform_kitti, read_all_lines
import cv2


class VIRTUALKITTIDataset2(Dataset):
    def __init__(self, datapath, list_filename, training, args,bs=None,bs_type=None):
        self.datapath = datapath
        self.bs = bs
        self.bs_type = bs_type
        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        if self.training:
            assert self.disp_filenames is not None

    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if len(splits[0]) == 2:  # ground truth not available
            return left_images, right_images, None
        else:
            if self.bs is not None: # this dataset is used in bootstrap
                print('this dataset is used in bootstrap')
                folder_name = os.path.join('bootstrap_data','vkitti2',self.bs_type,str(self.bs))
                # in prepare_bs.py, the saved bs gt is named after scene index (Scene01...) and left_filename
                disp_images = [os.path.join(folder_name,fn.split('/')[1]+'_'+fn.split('/')[-1]) for fn in left_images]
            else:
                disp_images = [x[2] for x in splits]
            return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        depth_gt = cv2.imread(filename, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        mask = (depth_gt > 0)
        disp = np.zeros(depth_gt.shape,'float32')
        B = 53.2725
        f = 725.0087
        disp[mask] = f*B/depth_gt[mask]
        return disp

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))

        if self.disp_filenames:  # has disparity ground truth
            if self.bs is None:
                disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
            else: # this dataset is used in bs
                disparity = self.load_disp(self.disp_filenames[index])

        else:
            disparity = None

        if self.training:
            w, h = left_img.size
            crop_w, crop_h = 512, 256

            x1 = random.randint(0, w - crop_w)
            y1 = random.randint(0, h - crop_h)

            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]

            # to tensor, normalize
            processed = get_transform_kitti()
            left_img = processed(left_img)
            right_img = processed(right_img)

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity}
        else:
            w, h = left_img.size # 1242 x 375

            # normalize
            processed = get_transform_kitti()
            left_img = processed(left_img).numpy()
            right_img = processed(right_img).numpy()


            # pad to size 1248x384
            top_pad = 384 - h
            right_pad = 1248 - w
            assert top_pad > 0 and right_pad > 0
            # pad images
            left_img = np.lib.pad(left_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)
            right_img = np.lib.pad(right_img, ((0, 0), (top_pad, 0), (0, right_pad)), mode='constant',
                                   constant_values=0)
            # pad disparity gt
            if disparity is not None:
                assert len(disparity.shape) == 2
                disparity = np.lib.pad(disparity, ((top_pad, 0), (0, right_pad)), mode='constant', constant_values=0)

            if disparity is not None:
                return {"left": left_img,
                        "right": right_img,
                        "disparity": disparity,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}
            else:
                return {"left": left_img,
                        "right": right_img,
                        "top_pad": top_pad,
                        "right_pad": right_pad,
                        "left_filename": self.left_filenames[index],
                        "right_filename": self.right_filenames[index]}
