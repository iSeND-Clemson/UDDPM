# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 23:35:43 2025

@author: MaxGr
"""

import os
import shutil
import random
from tqdm import tqdm

def sample_and_copy_images(src_dir, dst_dir, sample_ratio=0.1):
    """
    Uniformly sample images from a source directory and copy them to a destination directory.

    Args:
        src_dir (str): Path to the source directory containing images.
        dst_dir (str): Path to the destination directory where sampled images will be copied.
        sample_ratio (float): Ratio of images to sample (e.g., 0.1 for 10%).
    """
    # Ensure the source directory exists
    if not os.path.exists(src_dir):
        raise ValueError(f"Source directory '{src_dir}' does not exist.")

    # Create the destination directory if it doesn't exist
    os.makedirs(dst_dir, exist_ok=True)

    # Get a list of all files in the source directory
    all_files = [f for f in os.listdir(src_dir) if os.path.isfile(os.path.join(src_dir, f))]

    # Filter for image files based on extensions
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff'}
    image_files = [f for f in all_files if os.path.splitext(f)[1].lower() in valid_extensions]

    # Compute the number of images to sample
    num_samples = max(1, int(len(image_files) * sample_ratio))

    # Uniformly sample the images
    sampled_images = random.sample(image_files, num_samples)

    # Copy sampled images to the destination directory
    for image in tqdm(sampled_images):
        src_path = os.path.join(src_dir, image)
        dst_path = os.path.join(dst_dir, image)
        shutil.copy(src_path, dst_path)

    print(f"Copied {num_samples} images from '{src_dir}' to '{dst_dir}'.")

# Example usage
if __name__ == "__main__":
    # src_directory = "FLAME 2/no_fire"
    src_directory = "FLAME"
    
    dst_directory = "FLAME_sample"
    sample_ratio = 0.05  # Sample 10% of the images

    sample_and_copy_images(src_directory, dst_directory, sample_ratio)












