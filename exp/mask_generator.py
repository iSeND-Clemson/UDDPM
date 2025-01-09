# -*- coding: utf-8 -*-
"""
Created on Thu Aug 17 21:41:33 2023

@author: MaxGr
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

import cv2
import random
import numpy as np
from tqdm import tqdm
# import time
# import matplotlib.pyplot as plt

mask_path = './mask/'
os.makedirs(mask_path, exist_ok=True)


num_img = 1000
width, height = 512, 512  # Dimensions of the mask image
num_regions = random.randint(0, 5)  # Number of random regions (shapes) to generate
size = 10
max_size = 50


    
mask_info = {}
for i in tqdm(range(num_img)):
    mask = np.zeros((height, width), dtype=np.uint8)
    
    bbox_list = []
    j = 0
    while j < random.randint(1, 10):
        
        shape_type = np.random.choice(['rectangle', 'circle'], p=[0.5, 0.5])
        color = 255 #np.random.randint(0, 256)
        x, y = np.random.randint(size, width-size), np.random.randint(size, height-size)
        w, h = np.random.randint(size, max_size), np.random.randint(size, max_size)
        
        if w*h > 20000:
            continue
        else:
            j += 1
        
        if shape_type == 'rectangle':
            cv2.rectangle(mask, (x, y), (x + w, y + h), color, -1)
            x,y = round(x+w/2), round(y+h/2)
        elif shape_type == 'circle':
            # radius = np.random.randint(size, max_size)
            r = max(w,h)
            cv2.circle(mask, (x, y), r, color, -1)
            w,h = r+r,r+r
            
        bbox_list.append([x,y,w,h])
    for k in range(1):
        mask = cv2.blur(mask, (9,9))
        
    name = f"{i}.jpg"
    cv2.imwrite(os.path.join(mask_path, name), mask)
    mask_info[name] = bbox_list
    # .append([name, bbox_list])

# plt.imshow(mask)

import json
# from datetime import datetime
# current_datetime = datetime.now()
# date_time_string = current_datetime.strftime("%Y-%m-%d_%H-%M-%S")

with open('./mask_info.json', 'w') as f:
    json.dump(mask_info, f)

print('\n\n mask_info is saved as mask_info.json. Format: [x,y,w,h]')
# mask_info = np.array(mask_info, dtype=object)
# np.save('./wildfire_images/mask_info.npy',mask_info)





