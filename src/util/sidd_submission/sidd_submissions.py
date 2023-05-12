import numpy as np
import os
import argparse
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage import img_as_ubyte
import h5py
import scipy.io as sio
from pdb import set_trace as stx
import cv2

def get_files_in_directory(path, force_extension=None):
    assert os.path.isdir(path), f"{path} is not directory"
    file_paths = []
    # r=root, d=directories, f = files
    for r, d, f in os.walk(path):
        for file in f:
            if force_extension is not None:
                if force_extension in file:
                    file_paths.append(os.path.join(r, file))
            else:
                file_paths.append(os.path.join(r, file))
    return file_paths

#MPRNet?
parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='./output/x240/img/test_SIDD_benchmark_020_10-13-08-56-00/', type=str)
parser.add_argument('--result_dir', default='./output/x240/img/test_SIDD_benchmark_020_10-13-08-56-00/', type=str)
parser.add_argument('--eval_version', default='2.1.4', type=str)

args = parser.parse_args()

no_images = 40
patches_per_image = 32
c, h, w = 3, 256, 256
file_paths = get_files_in_directory(args.input_dir, '_DN.png')
restored = np.zeros([no_images, patches_per_image, h, w, c], dtype=np.uint8)
assert no_images * patches_per_image == len(file_paths), f"{no_images}x{patches_per_image} != {len(file_paths)}"

path_idx=0
with torch.no_grad():
    for i in tqdm(range(no_images)):
        for k in range(patches_per_image):
            restored_patch = cv2.imread(file_paths[path_idx], cv2.IMREAD_COLOR)#.transpose(2,1,0)
            restored_patch = cv2.cvtColor(restored_patch, cv2.COLOR_BGR2RGB)
            restored_patch = np.expand_dims(restored_patch, axis=0)
            path_idx += 1
            restored[i,k,:,:,:] = restored_patch

# save denoised data
sio.savemat(os.path.join(args.result_dir, 'SubmitSrgb.mat'), {"Idenoised": restored,'israw':False, 'eval_version':args.eval_version})
