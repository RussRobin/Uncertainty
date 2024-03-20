import numpy as np
import re
import torchvision.transforms as transforms


def get_transform(resize_w,resize_h):
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]


    return transforms.Compose([
        transforms.Resize((resize_h, resize_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

def get_transform_kitti():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def get_transform_sceneflow_train():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    crop_w, crop_h = 480, 256

    return transforms.Compose([
        transforms.Resize((int(crop_h/2),int(crop_w/2))),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

# def get_transform_sceneflow_test():
#     mean = [0.485, 0.456, 0.406]
#     std = [0.229, 0.224, 0.225]
#     crop_w, crop_h = 960, 512

#     return transforms.Compose([
#         transforms.Resize((int(crop_h/2),int(crop_w/2))),
#         transforms.ToTensor(),
#         transforms.Normalize(mean=mean, std=std),
#     ])
def get_transform_sceneflow_test():
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    crop_w, crop_h = 960, 512

    return transforms.Compose([
        transforms.CenterCrop((crop_h, crop_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

# read all lines in a file
def read_all_lines(filename):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    return lines


# read an .pfm file into numpy array, used to load SceneFlow disparity files
def pfm_imread(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale
