# -*- coding: utf-8 -*-
"""
Created on Wed Aug  2 01:25:16 2023

@author: MaxGr
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
TORCH_CUDA_ARCH_LIST="8.6"

import os
import cv2
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from glob import glob
from torchsummary import summary


import torch
import torch.nn as nn
import torch.nn.functional as F


print('torch.version: ',torch. __version__)
print('torch.version.cuda: ',torch.version.cuda)
print('torch.cuda.is_available: ',torch.cuda.is_available())
print('torch.cuda.device_count: ',torch.cuda.device_count())
print('torch.cuda.current_device: ',torch.cuda.current_device())
device_default = torch.cuda.current_device()
torch.cuda.device(device_default)
print('torch.cuda.get_device_name: ',torch.cuda.get_device_name(device_default))
device = torch.device("cuda")
# print(torch.cuda.get_arch_list())


# the architecture of the gan
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv_init = nn.Sequential( 
            nn.Conv2d(3, 32, 5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.MaxPool2d(2)
        )
        
        self.conv_1 = nn.Sequential(   
            nn.Conv2d(32, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.MaxPool2d(2)
        )
        
        self.conv_2 = nn.Sequential(   
            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.MaxPool2d(2)
        )
        
        self.conv_nonlinear = nn.Sequential(   
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 16, 3, stride=1, padding=1),
            nn.Tanh()
        )
        
        
        self.deconv_1 = nn.Sequential(
            nn.Conv2d(16, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            #nn.Upsample(scale_factor=2, mode='bilinear')
            nn.ConvTranspose2d(128, 128, 2, stride=2, padding=0, output_padding=0)
        )
        
        self.deconv_2 = nn.Sequential(
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            nn.Conv2d(64, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            #nn.Upsample(scale_factor=2, mode='bilinear')
            nn.ConvTranspose2d(64, 64, 2, stride=2, padding=0, output_padding=0)
        )
        
        self.deconv_3 = nn.Sequential(
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            
            #nn.Upsample(scale_factor=2, mode='bilinear')
            nn.ConvTranspose2d(32, 32, 2, stride=2, padding=0, output_padding=0)
        )
        
        self.deconv_4 = nn.Sequential(
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 32, 3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(32, 1, 3, stride=1, padding=1),
        )
        
    
    def forward(self,x):
        x = self.conv_init(x)
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = self.conv_nonlinear(x)
        
        x = self.deconv_1(x)
        x = self.deconv_2(x)
        x = self.deconv_3(x)
        # print(x.shape)
        x = self.deconv_4(x)
        # print(x.shape)
        return x
    
    
model = AE().to(device)
# output = model(x)
# print(output.size())  # Output size: torch.Size([1, 1, 512, 512])

device = torch.device("cuda")
summary(model, (3,512,512))



    

class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Contracting Path (Encoder)
        self.conv1 = self.conv_block(3, 32)
        self.conv2 = self.conv_block(32, 64)
        self.conv3 = self.conv_block(64, 128)
        self.conv4 = self.conv_block(128, 256)
        self.conv5 = self.conv_block(256, 512)

        # Expansive Path (Decoder)
        self.upconv6 = self.upconv_block(512, 256)
        self.conv6 = self.conv_block(512, 256)
        self.upconv7 = self.upconv_block(256, 128)
        self.conv7 = self.conv_block(256, 128)
        self.upconv8 = self.upconv_block(128, 64)
        self.conv8 = self.conv_block(128, 64)
        self.upconv9 = self.upconv_block(64, 32)
        self.conv9 = self.conv_block(64, 32)

        # Output layer
        self.output_layer = nn.Conv2d(32, 1, kernel_size=1)
        
        # Apply weight init
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=1.0)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )

    def upconv_block(self, in_channels, out_channels):
        return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)

    def forward(self, x):
        # Contracting Path
        x1 = self.conv1(x)
        x2 = F.max_pool2d(x1, kernel_size=2, stride=2)
        x3 = self.conv2(x2)
        x4 = F.max_pool2d(x3, kernel_size=2, stride=2)
        x5 = self.conv3(x4)
        x6 = F.max_pool2d(x5, kernel_size=2, stride=2)
        x7 = self.conv4(x6)
        x8 = F.max_pool2d(x7, kernel_size=2, stride=2)
        x9 = self.conv5(x8)

        # Expansive Path
        x = self.upconv6(x9)
        x = torch.cat([x, x7], dim=1)
        x = self.conv6(x)
        x = self.upconv7(x)
        x = torch.cat([x, x5], dim=1)
        x = self.conv7(x)
        x = self.upconv8(x)
        x = torch.cat([x, x3], dim=1)
        x = self.conv8(x)
        x = self.upconv9(x)
        x = torch.cat([x, x1], dim=1)
        x = self.conv9(x)

        # Output layer
        x = self.output_layer(x)

        return x





# Test the model
# if __name__ == "__main__":
x = torch.randn(1, 3, 512, 512)
model = UNet().to(device)
# output = model(x)
# print(output.size())  # Output size: torch.Size([1, 1, 512, 512])

device = torch.device("cuda")
summary(model, (3,512,512))











import os
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CustomDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.image_list = os.listdir(image_paths)
        self.mask_list = os.listdir(mask_paths)
        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_paths, self.image_list[idx])
        
        file_name = self.image_list[idx]
        image_name = file_name.split('_')[0] + '.jpg'

        mask_path  = os.path.join(self.mask_paths, image_name)
            
        # image_path = self.image_paths[idx]
        # mask_path = self.mask_paths[idx]

        # print(image_path)
        # print(mask_path)
        
        # Load image and mask using PIL (you can use other libraries if needed)
        image = Image.open(image_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Convert mask to grayscale (single channel)

        # Apply transformations if provided
        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        # Generate random factors for each layer
        random_factors = np.random.uniform(0, 1, size=3)
        for i in range(3):
            image[i, :, :] = image[i, :, :] * random_factors[i]
            
        # Convert to boolean tensor
        # mask_binary = (mask != 0).int()
        # mask = mask_binary.bool().float()
        # print(np.unique(mask))


        return image, mask



# Example data and targets (dummy data, replace with your actual data)
image_paths = './train/samples_diffusion/'
mask_paths  = './train/samples_mask/' 


# Transforms to be applied to the data
transform = transforms.Compose([
    # transforms.ToPILImage(),        # Convert to PIL Image
    # transforms.Resize((256, 256)),  # Resize to (256, 256)
    transforms.ToTensor()           # Convert to tensor
])

# Create custom dataset and dataloader
custom_dataset = CustomDataset(image_paths, mask_paths, transform)
train_loader = DataLoader(custom_dataset, batch_size=8, shuffle=True)

# Test the data loader
for batch in train_loader:
    images, masks = batch
    print(images.size())    # torch.Size([32, 3, 256, 256]) (batch_size=32, 3 channels, 256x256 size)
    print(masks.size())   # torch.Size([32, 1, 256, 256]) (batch_size=32, single-channel masks, 256x256 size)
    break


# for i in range(5):
#     plt.figure()
#     img = images[i]
#     img = img.permute(1,2,0)
#     plt.imshow(img)
#     plt.figure()
#     mask = masks[i]
#     mask = mask.permute(1,2,0)
#     plt.imshow(mask)
    
# temp = mask.data.numpy()[:,:,0]




def norm_image(x):
    # Get the minimum and maximum values in the tensor
    min_value = np.min(x)
    max_value = np.max(x)

    # Normalize the tensor to the range [0, 1]
    normalized_array = (x - min_value) / (max_value - min_value) *255

    return normalized_array.astype(np.uint8)


import cv2

def display_result(images, masks, outputs):
    # Select the first 3 images from the batch
    images = images[:4]
    masks = masks[:4]
    outputs = outputs[:4]

    images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    masks = masks.detach().cpu().permute(0, 2, 3, 1).numpy()
    outputs = outputs.detach().cpu().permute(0, 2, 3, 1).numpy()

    for i in range(4):
        image = images[i, :, :]
        mask = masks[i, :, :, 0]
        output = outputs[i, :, :, 0]
        
        heatmap_rgb = plt.get_cmap('jet')(output)[:, :, :3]  # RGB channels only

        mask = cv2.merge((mask, mask, mask))
        output = cv2.merge((output, output, output))

        # Normalize and convert to uint8
        image = norm_image(image)
        mask = norm_image(mask)
        output = norm_image(output)
        heatmap_rgb = norm_image(heatmap_rgb)
        heatmap_rgb = cv2.cvtColor(heatmap_rgb, cv2.COLOR_BGR2RGB)


        # Stack the mask and output horizontally
        combined_mask_output = cv2.hconcat([image, mask, output, heatmap_rgb])

        if i == 0:
            combined_image = combined_mask_output
        else:
            combined_image = cv2.vconcat([combined_image, combined_mask_output])
    
    return combined_image





import torch.optim as optim

# Move the model and loss function to the GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Move model and loss to GPU
model = AE().to(device)

# Define loss function (using Mean Squared Error loss)
criterion = nn.MSELoss().to(device)

# Define optimizer (you can adjust learning rate and other hyperparameters)
optimizer = optim.Adam(model.parameters(), lr=1e-3)



import time
from tqdm import tqdm


# Define the training loop
weight_file_path = 'best.pth'
if os.path.exists(weight_file_path):
    # Load the weights and resume training
    model.load_state_dict(torch.load(weight_file_path))
    print("Weight file exists. Resuming training...")

num_epochs = 100
loss_list = []





for epoch in range(num_epochs):
    start_time = time.time()  # Record the start time
    model.train()  # Set the model to training mode
    running_loss = 0.0
    pbar = tqdm(train_loader)
    
    for i, (images, masks) in enumerate(pbar):
    # for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        
        outputs = model(images)

        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        # Calculate elapsed time for the current epoch
        epoch_time = time.time() - start_time
        remaining_epochs = num_epochs - epoch - 1
        estimated_remaining_time = epoch_time * remaining_epochs
    
        # Update running loss
        running_loss += loss.item() * images.size(0)
        pbar.set_postfix(loss=loss.item(), Epochs=epoch)


    # Calculate average loss for the epoch
    # epoch_loss = running_loss / len(train_loader.dataset)
    epoch_loss = running_loss / i
    loss_list.append(epoch_loss)

    estimated_remaining_hour = estimated_remaining_time/3600
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, '
          f'Epoch Time: {epoch_time:.2f}s, '
          f'Estimated Remaining Time: {estimated_remaining_time:.2f}s ({estimated_remaining_hour:.2f}h)')
    
    # Save model weights after every epoch
    torch.save(model.state_dict(), f'last.pt')
    
    
    # Save some visualizations of the input images, predicted masks, and ground truth masks
    with torch.no_grad():
        outputs = model(images)
        
        combined_image = display_result(images, masks, outputs)

        plt.figure()
        plt.imshow(combined_image)
        plt.show()
        cv2.imwrite('./train/results/'+str(epoch)+'.png', combined_image)


# Calculate the total training time
total_training_time = time.time() - start_time
print('Finished Training')
print(f'Total Training Time: {total_training_time:.2f}s')


# Plot the loss curve
plt.plot(loss_list)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.show()














