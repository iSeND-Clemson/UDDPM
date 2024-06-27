# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 00:26:38 2024

@author: MaxGr
"""

import time
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

path = './'

dataset_info = np.load('dataset.npy', allow_pickle=True)


# image_info.append([i,
#                    image_save_name, 
#                    mask_save_name,
#                    opt.prompt, 
#                    noise_std, 
#                    opt.scale,
#                    opt.strength,
#                    opt.ddim_steps,
#                    R,G,B
#                    ])

for item in dataset_info:
    























