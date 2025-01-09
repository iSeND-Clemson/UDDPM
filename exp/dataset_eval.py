# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 22:56:40 2024

@author: MaxGr
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

import cv2
import time
import glob
import json
import random
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image
from scipy import linalg

import torch
import clip
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
    

# Function to load images from a directory and close files properly
def load_images_from_directory(directory, max_images=None):
    images = []
    file_list = os.listdir(directory)
    random.shuffle(file_list)

    for filename in tqdm(file_list, desc=f"Load images from {directory}"):
        if max_images and len(images) >= max_images: break
        if filename.endswith(".jpg") or filename.endswith(".png"):
            with Image.open(os.path.join(directory, filename)) as img:
                images.append(img.convert("RGB"))
                
    return images


# Function to extract features using Inception v3 with tqdm progress bar
def get_inception_features(images, model):
    model.eval()
    model.to(device)
    features = []
    with torch.no_grad():
        for img in tqdm(images, desc="Extracting features"):
            img = transform(img).unsqueeze(0).to(device)
            feat = model(img)[0].flatten().cpu().numpy()
            features.append(feat)
    return np.array(features)


# Function to calculate FID score
def calculate_fid(real_features, generated_features):
    mu1, sigma1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu2, sigma2 = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)
    ssdiff = np.sum((mu1 - mu2)**2.0)
    covmean = linalg.sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = ssdiff + np.trace(sigma1 + sigma2 - 2.0 * covmean)
    return fid



# data_folder = './exp/style_off/random/'
# data_folder = './exp/style_off/noise_mask/0/'
# data_folder = './exp/style_off/noise_mask/0.01/'
# data_folder = './exp/style_off/noise_mask/0.05/'
# data_folder = './exp/style_off/noise_mask/0.1/'
# data_folder = './exp/style_off/noise_mask/0.5/'
# data_folder = './exp/style_off/perlin/'
# data_folder = './exp/style_off/binary_mask/'
data_folder = './outputs/noise_test_0.1/'
data_folder = './outputs/perlin_test/'


# data_folder = './exp/style_on/random/'
# data_folder = './exp/style_on/noise_mask/0/'
# data_folder = './exp/style_on/noise_mask/0.01/'
# data_folder = './exp/style_on/noise_mask/0.05/'
# data_folder = './exp/style_on/noise_mask/0.1/'
# data_folder = './exp/style_on/noise_mask/0.5/'
# data_folder = './exp/style_on/perlin/'
# data_folder = './exp/style_on/binary_mask/'

# data_folder = './exp/perlin/'


image_folder = f'{data_folder}/samples/'
mask_folder = f'{data_folder}/masks/'
annotation_folder = f'{data_folder}/annotation/'
os.makedirs(annotation_folder, exist_ok=True)


image_info = np.load(f'{data_folder}/dataset.npy', allow_pickle=True)
mask_file = './mask_info.json'
# mask_info = np.load(mask_file, allow_pickle=True)
with open(mask_file, 'r') as f:
    mask_info = json.load(f)


prompt = "wildfire image"

# CLASSES = ["fire","non-fire"]
CLASSES = ["wildfire","non-fire"]

CLIP_score_list = []
CLIP_similarity_list = []
CLIP_conf_list = []

start_time = time.time()
file_list = os.listdir(image_folder)
for file in file_list:
    names = file.split('_')
    image_name, mask_name, index = names[-4:-1]
    if len(names)>4: image_name = f'{names[-5]}_{image_name}'
    
    image_path = image_folder+file
    # mask_path = f'{mask_folder}/{image_name}_{mask_name}_{index}.png'
    # mask_path = f'{mask_folder}/{image_name}_{mask_name}_{index}_mask.png'
    mask_path = f'{mask_folder}/{image_name}_{mask_name}_{index}.jpg'

    image_i = Image.open(image_path)
    mask_i = Image.open(mask_path).convert("L")

    image_annotation = np.array(image_i)
    mask_annotation = np.array(mask_i)#.astype(bool)
    
    
    # # Find contours in the mask, or use the mask_info.json
    # for i in range(5):
    # #     mask_annotation = cv2.medianBlur(mask_annotation, 7)
    #      mask_annotation = cv2.bilateralFilter(mask_annotation, 13, 128, 128)

    # th = 127
    # mask_annotation[mask_annotation>th]=255
    # mask_annotation[mask_annotation<=th]=0
    
    # contours, _ = cv2.findContours(mask_annotation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # mask_annotation = cv2.cvtColor(mask_annotation, cv2.COLOR_GRAY2BGR)
    # if contours==(): continue

    # boxes = []
    # # # Draw bounding boxes around each contour
    # for contour in contours:
    #     x, y, w, h = cv2.boundingRect(contour)
    #     boxes.append([x,y,w,h])
        
        
    boxes = mask_info[mask_name +'.jpg']    
    boxes = np.array(boxes)

    largest = 10
    areas = boxes[:, 2] * boxes[:, 3]
    sorted_indices = np.argsort(areas)[::-1]
    largest_boxes = boxes[sorted_indices[:largest]]
    
    clip_list = []
    for i in largest_boxes:
        x,y,w,h = i
        x1, y1 = int(x-w/2), int(y-h/2)
        x2, y2 = int(x+w/2), int(y+h/2)
        x1,y1,x2,y2 = max(x1,0),max(y1,0),max(x2,0),max(y2,0)
        if w*h < 50: continue
        cropped_area = image_annotation[y1:y2, x1:x2].copy()
        cv2.rectangle(image_annotation,(x1,y1), (x2,y2), (255,0,0), 2)
        # cv2.rectangle(mask_annotation,(x1,y1), (x2,y2), (255,0,0), 2)
        clip_list.append([x,y,cropped_area])
        
    # plt.imshow(mask_annotation)   
    # plt.imshow(image_annotation)
    # plt.imshow(cropped_area)   
    
    # clip_area = preprocess(Image.fromarray(cropped_area)).unsqueeze(0).to(device)
    image = preprocess(image_i).unsqueeze(0).to(device)
    
    text = prompt.strip()
    text = clip.tokenize([text]).to(device)
    text_class = clip.tokenize(CLASSES).to(device)

    with torch.no_grad():
        image_feature = model.encode_image(image)
        text_feature = model.encode_text(text)
        # mask_features = model.encode_image(mask)

        for x,y,cropped_area in clip_list:
            clip_area = preprocess(Image.fromarray(cropped_area)).unsqueeze(0).to(device)
    
            logits_per_image, logits_per_text = model(clip_area, text_class)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
            
            class_id = 0 #probs[0].argmax()
            conf = probs[0][class_id]
            # print(f"{CLASSES[class_id]},{conf:0.2f},{w*h}")
            CLIP_conf_list.append(conf)
            cv2.putText(image_annotation, f'{conf:0.2f}', (x,y-5), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        cv2.imwrite(f'{annotation_folder}/{file}', cv2.cvtColor(image_annotation, cv2.COLOR_RGB2BGR))

    
    similarities = (image_feature @ text_feature.T).diag().cpu().numpy()
    # image_similarity = torch.cosine_similarity(image_feature, mask_features).cpu().numpy()

    # Calculate CLIP score
    clip_score = similarities.mean()
    # similarity_score = image_similarity.mean()
    # print(f'{file} | CLIP Score: {clip_score} | Similarity Score: {similarity_score}')
    print(f'{file} | CLIP Score: {clip_score} | CLIP Conf: {conf}')

    CLIP_score_list.append(clip_score)
    # CLIP_similarity_list.append(similarity_score)
    
end_time = time.time()
print(f'Total tine cost: {end_time-start_time}')

# valid_data = np.array(valid_data, dtype=object)
save_folder = './CLIP_test/'
# np.save(f'{save_folder}CLIP_data.npy', dataset_info)
# plt.plot(CLIP_score_list)
# plt.plot(CLIP_similarity_list)
# plt.scatter(CLIP_similarity_list, CLIP_score_list)

clip_score = np.mean(CLIP_score_list)
clip_conf = np.mean(CLIP_conf_list)
similarity_score = np.mean(CLIP_similarity_list)

print(f'CLIP_score: {clip_score} | CLIP_similarity: {similarity_score} | CLIP_conf: {clip_conf}')




fid_score_list = []

# Load Inception v3 model
inception_model = models.inception_v3(pretrained=True, transform_input=False)

# Directory paths for real and generated images
real_features = 'real_features.npy'
if os.path.exists(real_features):
    real_features = np.load(real_features, allow_pickle=True)
else:
    real_images_dir = '../wildfire_images/FLAME 2/fire_smoke/'
    # real_images_dir = '../wildfire_images/FLAME/'
    real_images = load_images_from_directory(real_images_dir)
    real_features = get_inception_features(real_images, inception_model)
    # np.save('real_features.npy', real_features)

    
# Load generated images (specify max_images if needed)
generated_images_dir = image_folder
generated_images = load_images_from_directory(generated_images_dir)
generated_features = get_inception_features(generated_images, inception_model)

# Calculate FID
fid_score = calculate_fid(real_features, generated_features)
fid_score_list.append(fid_score)

print('FID Score:', fid_score)
print(clip_score, similarity_score, fid_score)























