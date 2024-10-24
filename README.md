
# FLAME Diffuser: Wildfire Image Synthesis using Mask Guided Diffusion



## Introduction

Wildfires have devastating impacts on natural environments and human settlements. Existing fire detection systems rely on large, annotated datasets that often lack geographic diversity, leading to decreased generalizability. To address this, we introduce the **FLAME Diffuser**, a diffusion-based framework that synthesizes high-quality wildfire images with precise flame location control. This training-free framework eliminates the need for model fine-tuning, enhancing the development of robust wildfire detection models.


## Sample Dataset

<img src="./Figure/grid.jpg" width=100%>

- **Dataset:** Download from [Google Drive](https://drive.google.com/drive/folders/1Brt5TvkdTUqJPGtXSLGQNCc3kgk2NygD?usp=sharing)

---


## Key Features

- **Training-Free Diffusion Framework:** Generates wildfire images without the need for model training or fine-tuning.
- **Precise Flame Control:** Utilizes noise-based masks for accurate flame placement.
- **Diverse Backgrounds:** Creates images with varied and realistic backgrounds, enhancing model generalizability.
- **High-Quality Dataset:** Introduces FLAME-SD with 10,000 synthesized images for robust model training.

## Methodology

<img src="./Figure/diffuser_2.jpg" width=90%>

1. **Mask Generation:** 
   - Masks are generated to define areas for fire elements using fundamental shapes like rectangles and circles.
   - Perlin noise is added to the masks to create a smoother integration process.

2. **Diffusion Process:**
   - Combines masks with raw images, processed through a Variational Autoencoder (VAE) to generate latent variables.
   - The denoising U-Net refines these variables to produce realistic images guided by text prompts.


## Experimental Results

<img src="./Figure/fid.jpg" width=90%>

- **High-Quality:** Lowest FID score compared to other methods, indicating better realistic styles.
- **Consistency:** Image content does not shift, the semantic information is well-kept in synthesized images.


<img src="./Figure/table.jpg" width=70%>

For more details, visit the [Project Page](https://arazi2.github.io/aisends.github.io/project/flame).



# Citation 
<a href="https://arxiv.org/abs/2403.03463">FLAME Diffuser: Wildfire Image Synthesis using Mask Guided Diffusion</a>

@article{wang2024flame,
  title={FLAME Diffuser: Grounded Wildfire Image Synthesis using Mask Guided Diffusion},
  author={Wang, Hao and Boroujeni, Sayed Pedram Haeri and Chen, Xiwen and Bastola, Ashish and Li, Huayu and Razi, Abolfazl},
  journal={arXiv preprint arXiv:2403.03463},
  year={2024}
}
