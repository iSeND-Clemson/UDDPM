# -*- coding: utf-8 -*-
"""
Created on Sat Jul 13 03:48:41 2024

@author: MaxGr
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

import cv2
import numpy as np
# import time
# import random
# import matplotlib.pyplot as plt


def lerp(a, b, x):
    return a + x * (b - a)

def fade(t):
    return 6 * t**5 - 15 * t**4 + 10 * t**3

def gradient(h, x, y):
    vectors = np.array([[0, 1], [0, -1], [1, 0], [-1, 0]])
    g = vectors[h % 4]
    return g[:, :, 0] * x + g[:, :, 1] * y

def perlin(x, y, seed=0):
    # np.random.seed(seed)
    p = np.arange(512, dtype=int)
    np.random.shuffle(p)
    p = np.stack([p, p]).flatten()
    xi = x.astype(int)
    yi = y.astype(int)
    xf = x - xi
    yf = y - yi
    u = fade(xf)
    v = fade(yf)
    n00 = gradient(p[p[xi] + yi], xf, yf)
    n01 = gradient(p[p[xi] + yi + 1], xf, yf - 1)
    n11 = gradient(p[p[xi + 1] + yi + 1], xf - 1, yf - 1)
    n10 = gradient(p[p[xi + 1] + yi], xf - 1, yf)
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)
    return lerp(x1, x2, v)

def img_uint8(image):
    image = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
    image = image.astype(np.uint8)
    return image



from PIL import Image
from tqdm import tqdm

path = './perlin_mask/'
os.makedirs(path, exist_ok=True)

std_devs = np.arange(0, 100)

mask_path = './mask/'
mask_list = os.listdir(mask_path)
for mask_i in tqdm(mask_list):
    name_id = mask_i.split('.')[0]
    mask = Image.open(mask_path+mask_i).convert("L")
    mask = np.array(mask)
    
    # Perlin
    lin = np.linspace(0, int(name_id)/10, 512, endpoint=False)
    x, y = np.meshgrid(lin, lin)
    noise = perlin(x, y)#, seed=round(std*1000))
    # noise = cv2.cvtColor(img_uint8(noise), cv2.COLOR_GRAY2RGB)/255
    noisy_image = noise
    noisy_image = img_uint8(noisy_image)/255.0

    perlin_mask = noisy_image*mask
    for k in range(1):
        perlin_mask = cv2.blur(perlin_mask, (9,9))
    
    cv2.imwrite(os.path.join(path, mask_i), perlin_mask)









