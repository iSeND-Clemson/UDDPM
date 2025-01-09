# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 01:02:48 2023

@author: MaxGr
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

# import sys
# from utils import *


import torch
import clip
from PIL import Image

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)



# from transformers import CLIPProcessor, CLIPModel
# model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")



def Flame_ratio(image, box):
    x1 = box[0]
    y1 = box[1]
    x2 = box[2]
    y2 = box[3]

    # print(box)
    zone = image[y1:y2,x1:x2]
    w,h = zone.shape
    # plt.imshow(zone)
    
    flame_ratio = np.sum(zone[zone>0])/((w*h)*255)
    # flame_ratio = flame_ratio*np.sum(zone[zone>0])
    return flame_ratio




def NMS(image, boxes, zone_overlap_thred, num_areas, flame_thred):
    # Return an empty list, if no boxes given
    if len(boxes) == 0:
        return []
    else:
        boxes[:, 2] = boxes[:, 2]+boxes[:, 0]
        boxes[:, 3] = boxes[:, 3]+boxes[:, 1]
                
    x1 = boxes[:, 0]  # x coordinate of the top-left corner
    y1 = boxes[:, 1]  # y coordinate of the top-left corner
    x2 = boxes[:, 2]  # x coordinate of the bottom-right corner
    y2 = boxes[:, 3]  # y coordinate of the bottom-right corner
    # Compute the area of the bounding boxes and sort the bounding
    # Boxes by the bottom-right y-coordinate of the bounding box
    areas = (x2 - x1 + 1) * (y2 - y1 + 1) # We add 1, because the pixel at the start as well as at the end counts

    areas_order = np.argsort(areas)[::-1]
    # areas_list = areas[areas_order]
    boxes = boxes[areas_order]

    # The indices of all boxes at start. We will redundant indices one by one.
    indices = np.arange(len(x1))
    
    for i,box in enumerate(boxes): #print(i,box)
        # Create temporary indices  
        temp_indices = indices[indices!=i]
        # Find out the coordinates of the intersection box
        xx1 = np.maximum(box[0], boxes[temp_indices,0])
        yy1 = np.maximum(box[1], boxes[temp_indices,1])
        xx2 = np.minimum(box[2], boxes[temp_indices,2])
        yy2 = np.minimum(box[3], boxes[temp_indices,3])
        # Find out the width and the height of the intersection box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / areas[temp_indices]
        # if the actual boungding box has an overlap bigger than treshold with any other box, remove it's index  
        if np.any(overlap) > zone_overlap_thred:
            indices = indices[indices != i]
            
    #return only the boxes at the remaining indices
    # areas = areas[indices].astype(int)
    boxes = boxes[indices].astype(int)
        
    flame_list = []
    for i in range(len(boxes)):
        box = boxes[i]
        flame_ratio = Flame_ratio(image, box)
        flame_list.append(flame_ratio)
    flame_list = np.array(flame_list)
    flame_order = np.argsort(flame_list)[::-1]
    flame_list = flame_list[flame_order]
    boxes = boxes[flame_order]

    # print(flame_list)
    # print(boxes)
    boxes = np.delete(boxes, np.where(flame_list < flame_thred), 0)


    boxes[:, 2] = boxes[:, 2]-boxes[:, 0]
    boxes[:, 3] = boxes[:, 3]-boxes[:, 1]
    
    # max_flame = Nlargest(flame_list, num_areas)[1]
    # boxes = boxes[max_flame]
    
    boxes = boxes[0:num_areas]
    return boxes




import time
import glob
import random
import numpy as np
import matplotlib.pyplot as plt
import cv2

path = './test_data/'

dataset_info = np.load(path + 'dataset.npy', allow_pickle=True)

# dataset_info = np.load('dataset.npy', allow_pickle=True)


image_folder = path + '/flame_mask_1/'
mask_folder  = path + '/mask/'







# CLASSES = ["fire"]
CLASSES = ["fire","smoke", "tree", "rock", "people", "building", "car", "cloud", "snow"]






zone_overlap_thred = 0.8
flame_thred = 0.2
num_areas = 300

image_CLIP = []
valid_images = []
valid_data = []
rgb_values = []

stat = []

start_time = time.time()
for i in range(len(dataset_info)):
    image_info = dataset_info[i]
    
    image_name = image_info[1]
    mask_name  = image_info[2]
    
    image_path = image_folder+image_name
    mask_path  = mask_folder+mask_name 

    print(image_name)
    
    image_bgr = cv2.imread(image_path)
    # image_bgr = resize_image(image_bgr, 1024)
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    # plt.imshow(image_rgb)
    
    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    boxes = []
    # # Draw bounding boxes around each contour
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        boxes.append([x,y,w,h])
    
    boxes = np.array(boxes)

    largest = 1
    areas = boxes[:, 2] * boxes[:, 3]
    sorted_indices = np.argsort(areas)[::-1]
    largest_boxes = boxes[sorted_indices[:largest]]
    
    mask  = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    for i in largest_boxes:
        x,y,w,h = i
        # cv2.rectangle(image, (x,y), (x+w,y+h), (0,255,255), 2)
        # cv2.rectangle(seg_blur,  (x,y),(x+w,y+h), (255,255,255), 2)
        cv2.rectangle(mask,(x,y), (x+w,y+h), (0,0,255), 2)
    
    # plt.imshow(mask)
    
        
    bounding_boxes = largest_boxes
    # bounding_boxes = detections.xyxy
    
    bboxes = []
    classification_results = []
    for box in bounding_boxes:
        # x1, y1, x2, y2 = box.astype(int)
        
        x,y,w,h = box
        
        # w, h = int(w * image_rgb.shape[1]), int(h * image_rgb.shape[0])
        x1, y1 = int(x), int(y)
        x2, y2 = int(x+w), int(y+h)
    
        bboxes.append([x1, y1, x2, y2])
        cropped_area = image_rgb[y1:y2, x1:x2]
        # plt.imshow(cropped_area)
        r = np.max(cropped_area[:,:,0])
        g = np.max(cropped_area[:,:,1])
        b = np.max(cropped_area[:,:,2])
        rgb = [r,g,b]
        
        image = preprocess(Image.fromarray(cropped_area)).unsqueeze(0).to(device)
        text = clip.tokenize(CLASSES).to(device)
        
        with torch.no_grad():
            image_features = model.encode_image(image)
            text_features = model.encode_text(text)
            
            logits_per_image, logits_per_text = model(image, text)
            probs = logits_per_image.softmax(dim=-1).cpu().numpy()
        
        # print("Label probs:", probs)  # prints: [[0.9927937  0.00421068 0.00299572]]
        
        class_id = probs[0].argmax()
        confidence = probs[0][class_id]
    
        classification_results.append(f"{CLASSES[class_id]},{confidence:0.2f}")
        
        image_CLIP.append([CLASSES[class_id], confidence])
        
        if CLASSES[class_id] == 'fire':
            valid_images.append(image_name)
            valid_info = image_info.tolist()
            valid_info.extend([CLASSES[class_id], confidence, rgb])
            valid_data.append(valid_info)
            rgb_values.append(rgb)

            cv2.imwrite(f'./{path}/valid_images/{image_name}', image_bgr)
            
    temp = image_info.tolist()
    temp.extend([CLASSES[class_id], confidence, rgb])
    stat.append(temp)
    
        
end_time = time.time()
print(f'Total tine cost: {end_time-start_time}')


valid_data = np.array(valid_data, dtype=object)

np.save('./test_data/valid_data.npy',np.array(valid_data, dtype=object))









import cv2
import numpy as np
import matplotlib.pyplot as plt

def grayscale_to_spectrum(grayscale_value):
    """Maps a grayscale value (0-255) to a color in the RGB spectrum."""

    hue = (grayscale_value / 255) * 180  # Hue range 0-180 (half the HSV circle)
    saturation = 255  # Full saturation
    value = 255       # Full value (brightness)

    # Convert HSV to BGR (OpenCV format)
    bgr = cv2.cvtColor(np.uint8([[[hue, saturation, value]]]), cv2.COLOR_HSV2BGR)[0][0]

    return tuple(bgr)  # Convert to tuple (B, G, R)

# Example usage
gray_values = np.arange(0, 256, 5)  # Grayscale values from 0 to 255
color_band = np.array([grayscale_to_spectrum(v) for v in gray_values], dtype=np.uint8)
color_band = color_band.reshape((1, -1, 3))  # Reshape for display

#Display colors
plt.figure(figsize=(10, 5))
plt.imshow(color_band)
plt.axis('off')
plt.savefig(f'{path}/RGB.png')  # Saves the plot as a PNG file
plt.show()



data = valid_data[:,8:]


# rgb_values = [(float(r), float(g), float(b)) for r, g, b, _, _ in data]
new_rgb_values = [np.array(rgb)/255.0 for rgb in data[:,-1]]
confidences = [float(confidence) for _, _, _, _, confidence, _ in data]
# color_values = [0.2989 * r + 0.5870 * g + 0.1140 * b for r, g, b in rgb_values]
color_values = []

for item in data:
    color = item[:3]
    color = np.array(color).astype(float)*255
    
    distances = np.linalg.norm(color_band - color, axis=2)  # Calculate distances along axis=2 (RGB channels)
    index = np.argmin(distances)

    distances[0][index]
    color_band[0][index]
    color_values.append(index)

# Create a scatter plot
plt.figure(dpi=300, figsize=(5, 3))  # Adjust the figure size as needed
plt.scatter(color_values, confidences, c=new_rgb_values, cmap='RGB')
# plt.barh(confidences, color_values, color=rgb_values)
# plt.colorbar(label='RGB Color')
plt.xlabel('Color spectrum')
plt.ylabel('Confidence')  # You can adjust the label as needed

# Show the plot
plt.title('CLIP Score of Bounding Boxes')
plt.savefig(f'{path}/color_values.png')  # Saves the plot as a PNG file
plt.show()








# import colorsys

# data = valid_data[:,8:]



# # rgb_values = [(float(r), float(g), float(b)) for r, g, b, _, _ in data]
# new_rgb_values = [np.array(rgb)/255.0 for rgb in rgb_values]
# confidences = [float(confidence) for _, _, _, _, confidence in data]
# # color_values = [0.2989 * r + 0.5870 * g + 0.1140 * b for r, g, b in rgb_values]

# # hsv = []
# # for r,g,b in rgb_values:
# #     # r, g, b = r / 255.0, g / 255.0, b / 255.0

# #     h, s, v = colorsys.rgb_to_hsv(r,g,b)
# #     hsv.append(h)
    
# # hsv = np.array(hsv)



# # Create a scatter plot
# plt.figure(dpi=300, figsize=(10, 6))  # Adjust the figure size as needed
# plt.scatter(color_values, confidences, c=rgb_values, cmap='RGB')
# # plt.barh(confidences, color_values, color=rgb_values)
# # plt.colorbar(label='RGB Color')
# plt.xlabel('Color values')
# plt.ylabel('Confidence')  # You can adjust the label as needed

# # Show the plot
# plt.title('Scatter Plot of Confidence Colored by RGB')
# plt.savefig('./CLIP/plot/color_values.png')  # Saves the plot as a PNG file
# plt.show()




# hsv = []
# for r,g,b in rgb_values:
#     # r, g, b = r / 255.0, g / 255.0, b / 255.0

#     h, s, v = colorsys.rgb_to_hsv(r,g,b)
#     hsv.append(h)
    
# hsv = np.array(hsv)

# # Create a scatter plot
# plt.figure(dpi=300, figsize=(10, 6))  # Adjust the figure size as needed
# plt.scatter(hsv, confidences, c=new_rgb_values, cmap='RGB')
# # plt.barh(confidences, color_values, color=rgb_values)
# # plt.colorbar(label='RGB Color')
# plt.xlabel('Color values')
# plt.ylabel('Confidence')  # You can adjust the label as needed

# # Show the plot
# plt.title('Scatter Plot of Confidence Colored by RGB')
# plt.savefig('./CLIP/plot/hsv_values.png')  # Saves the plot as a PNG file
# plt.show()





data = valid_data[:,8:]

rgb_values = [(float(r), float(g), float(b)) for r, g, b, _, _, _ in data]
confidences = [float(confidence) for _, _, _, _, confidence, _ in data]
# color_values = [0.2989 * r + 0.5870 * g + 0.1140 * b for r, g, b in rgb_values]


# Create a scatter plot
plt.figure(dpi=200, figsize=(10, 6))  # Adjust the figure size as needed
plt.scatter(color_values, confidences, c=rgb_values, cmap='RGB')
# plt.barh(confidences, color_values, color=rgb_values)
# plt.colorbar(label='RGB Color')
plt.xlabel('Color spectrum')
plt.ylabel('Confidence')  # You can adjust the label as needed

# Show the plot
plt.title('Mask color')
plt.savefig(f'{path}/diffusion_color_values.png')  # Saves the plot as a PNG file
plt.show()










stat = np.array(stat, dtype=object)

class_rgb = [np.array(rgb)/255.0 for rgb in stat[:,-1]]

plt.scatter(stat[:,11], np.arange(len(stat)), c=class_rgb, cmap='RGB')
# plt.scatter(color_values, confidences, c=new_rgb_values, cmap='RGB')


























