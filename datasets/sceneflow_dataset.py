import os
import random
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from datasets.data_io import get_transform, get_transform_sceneflow_test,read_all_lines, pfm_imread
import scipy.ndimage as ndi

class SceneFlowDatset(Dataset):
    def __init__(self, datapath, list_filename, training, args,bs=None,bs_type=None):
        self.datapath = datapath
        self.bs = bs
        self.bs_type = bs_type

        self.left_filenames, self.right_filenames, self.disp_filenames = self.load_path(list_filename)
        self.training = training
        try:
            self.zoom = args.zoom
            self.crop_w = args.crop_w
            self.crop_h = args.crop_h
        except:
            print('default zoom=1, crop_w = 448, crop_h = 284')
            self.zoom = 1
            self.crop_w = 448
            self.crop_h = 284


    def load_path(self, list_filename):
        lines = read_all_lines(list_filename)
        splits = [line.split() for line in lines]
        left_images = [x[0] for x in splits]
        right_images = [x[1] for x in splits]
        if self.bs is not None: # this dataset is used in bootstrap
            print('this dataset is used in bootstrap')
            folder_name = os.path.join('bootstrap_data','sceneflow',self.bs_type,str(self.bs))
            # check prepare_bs.py, how the saved bs gt is named
            disp_images = [os.path.join(folder_name,fn.split('/')[2]+'_' + fn.split('/')[3]+'_'+fn.split('/')[-1]) if len(fn.split('/')) == 5 else os.path.join(folder_name,fn.split('/')[2] + '_' + fn.split('/')[3]+'_'+ fn.split('/')[4]+'_'+fn.split('/')[-1]) for fn in left_images]
        else:
            disp_images = [x[2] for x in splits]
        return left_images, right_images, disp_images

    def load_image(self, filename):
        return Image.open(filename).convert('RGB')

    def load_disp(self, filename):
        if self.bs is None:
            data, scale = pfm_imread(filename)
            data = np.ascontiguousarray(data, dtype=np.float32)
        else:
            data = Image.open(filename)
            data = np.array(data, dtype=np.float32) / 256.
            print('data',data.shape)
        return data

    def __len__(self):
        return len(self.left_filenames)

    def __getitem__(self, index):
        left_img = self.load_image(os.path.join(self.datapath, self.left_filenames[index]))
        right_img = self.load_image(os.path.join(self.datapath, self.right_filenames[index]))
        if self.bs is None:
            disparity = self.load_disp(os.path.join(self.datapath, self.disp_filenames[index]))
        else: # this dataset is used in bs
            disparity = self.load_disp(self.disp_filenames[index])

        if self.training:
            w, h = left_img.size # (960,540) or (960,512)(if self.bs)
            crop_w, crop_h = w - self.crop_w, h - self.crop_h
            # crop_w, crop_h = 512, 256
            if self.bs is None:
                x1 = random.randint(0, w - crop_w)
                y1 = random.randint(0, h - crop_h)
            else: # in bs
                x1 = random.randint(0, w - crop_w)
                y1 = random.randint(14, min(h - crop_h,270))
            # random crop
            left_img = left_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            right_img = right_img.crop((x1, y1, x1 + crop_w, y1 + crop_h))
            if self.bs is None:
                disparity = disparity[y1:y1 + crop_h, x1:x1 + crop_w]
            else: # in bs
                disparity = disparity[y1-14:y1 + crop_h-14, x1:x1 + crop_w]

            # downsample to H/2, W/2, to tensor, normalize
            resize_w, resize_h = int(crop_w * self.zoom), int(crop_h * self.zoom)
            processed = get_transform(resize_w, resize_h)
            left_img = processed(left_img)
            right_img = processed(right_img)
            disparity = ndi.zoom(disparity, zoom=self.zoom)
            disparity = disparity * self.zoom

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "left_filename":self.left_filenames[index]}
        else:
            w, h = left_img.size
            resize_w, resize_h = int(w * self.zoom), int(h * self.zoom)
            # processed = get_transform(resize_w, resize_h)
            processed = get_transform_sceneflow_test()
            left_img = processed(left_img)
            right_img = processed(right_img)
            disparity = ndi.zoom(disparity, zoom=self.zoom)
            disparity = disparity * self.zoom

            # CWX NOTE
            crop_w, crop_h = 960, 512
            top = (h - crop_h) // 2
            left = (w - crop_w) // 2
            bottom = top + crop_h
            right = left + crop_w
           
            disparity = disparity[top:bottom, left:right]

            return {"left": left_img,
                    "right": right_img,
                    "disparity": disparity,
                    "top_pad": 0,
                    "right_pad": 0,
                    "left_filename":self.left_filenames[index]}
