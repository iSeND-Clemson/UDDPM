# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 11:56:55 2024

@author: MaxGr
"""


import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"
import time
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image

mask_folder = f'./perlin_mask/'

target_mask = cv2.imread('./exp/36561_489_mask.jpg')

file_list = os.listdir(mask_folder)

images = []
for file in file_list:
    print(file)
    mask_i = cv2.imread(mask_folder+file)
    diff = np.sum(mask_i-target_mask)
    images.append(diff)
    
    
images = np.array(images)

index = np.argmin(images)
    
print(file_list[index])
    