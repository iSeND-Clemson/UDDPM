# -*- coding: utf-8 -*-
"""
Created on Mon Aug 28 16:57:54 2023

@author: MaxGr
"""
import os
import shutil
from tqdm import tqdm


# img_path = './train/samples_mask/'
img_path = 'D:/Data/Flame 2/254p Dataset/254p RGB Images/'
save_path = './wildfire_images/FLAME 2/no_fire/'

for path in [img_path, save_path]:
    if not os.path.exists(path):
        os.makedirs(path)
        
        
img_list = os.listdir(img_path)

for image in tqdm(img_list):
    # img_file = random.choice(img_list)
    # img_file = img_list[i]
    image_name = image.split('.')[0]
    ID = int(image_name)
    if (1<= ID and ID <=13700):
        y = 0
        shutil.copy2(img_path+image, save_path+image)

        continue
    
        
    elif   (13701	<= ID and ID <=14699) \
        or (16226	<= ID and ID <=19802) \
        or (19900	<= ID and ID <=27183) \
        or (27515	<= ID and ID <=31294) \
        or (31510	<= ID and ID <=33597) \
        or (33930	<= ID and ID <=36550) \
        or (38031	<= ID and ID <=38153) \
        or (43084	<= ID and ID <=45279) \
        or (51207	<= ID and ID <=52286):
            
        y = 1
        continue
    
    else:
        # print(image)
        y=2
        # shutil.copy2(img_path+image, save_path+image)

        
        
        
