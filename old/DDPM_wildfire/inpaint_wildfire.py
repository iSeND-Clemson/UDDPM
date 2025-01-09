import sys
import numpy as np
import streamlit as st
from PIL import Image
from omegaconf import OmegaConf
from einops import repeat
from main import instantiate_from_config
from streamlit_drawable_canvas import st_canvas
import torch


from ldm.models.diffusion.ddim import DDIMSampler


MAX_SIZE = 640

# load safety model
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from transformers import AutoFeatureExtractor
# from imwatermark import WatermarkEncoder
import cv2

safety_model_id = "CompVis/stable-diffusion-safety-checker"
safety_feature_extractor = AutoFeatureExtractor.from_pretrained(safety_model_id)
safety_checker = StableDiffusionSafetyChecker.from_pretrained(safety_model_id)
wm = "StableDiffusionV1-Inpainting"
# wm_encoder = WatermarkEncoder()
# wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

wm_encoder = None

def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images

def put_watermark(img):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img

def check_safety(x_image):
    safety_checker_input = safety_feature_extractor(numpy_to_pil(x_image), return_tensors="pt")
    x_checked_image, has_nsfw_concept = safety_checker(images=x_image, clip_input=safety_checker_input.pixel_values)
    assert x_checked_image.shape[0] == len(has_nsfw_concept)
    return x_checked_image, has_nsfw_concept


@st.cache(allow_output_mutation=True)
def initialize_model(config, ckpt):
    config = OmegaConf.load(config)
    model = instantiate_from_config(config.model)

    model.load_state_dict(torch.load(ckpt)["state_dict"], strict=False)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)
    sampler = DDIMSampler(model)

    return sampler


def make_batch_sd(
        image,
        mask,
        txt,
        device,
        num_samples=1):
    image = np.array(image.convert("RGB"))
    image = image[None].transpose(0,3,1,2)
    image = torch.from_numpy(image).to(dtype=torch.float32)/127.5-1.0

    mask = np.array(mask.convert("L"))
    mask = mask.astype(np.float32)/255.0
    mask = mask[None,None]
    mask[mask < 0.5] = 0
    mask[mask >= 0.5] = 1
    mask = torch.from_numpy(mask)

    masked_image = image * (mask < 0.5)

    batch = {
            "image": repeat(image.to(device=device), "1 ... -> n ...", n=num_samples),
            "txt": num_samples * [txt],
            "mask": repeat(mask.to(device=device), "1 ... -> n ...", n=num_samples),
            "masked_image": repeat(masked_image.to(device=device), "1 ... -> n ...", n=num_samples),
            }
    return batch


def inpaint(sampler, image, mask, prompt, seed, scale, ddim_steps, num_samples=1, w=512, h=512):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = sampler.model

    prng = np.random.RandomState(seed)
    start_code = prng.randn(num_samples, 4, h//8, w//8)
    start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)

    with torch.no_grad():
        with torch.autocast("cuda"):
            batch = make_batch_sd(image, mask, txt=prompt, device=device, num_samples=num_samples)

            c = model.cond_stage_model.encode(batch["txt"])

            c_cat = list()
            for ck in model.concat_keys:
                cc = batch[ck].float()
                if ck != model.masked_image_key:
                    bchw = [num_samples, 4, h//8, w//8]
                    cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                else:
                    cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
                c_cat.append(cc)
            c_cat = torch.cat(c_cat, dim=1)

            # cond
            cond={"c_concat": [c_cat], "c_crossattn": [c]}

            # uncond cond
            uc_cross = model.get_unconditional_conditioning(num_samples, "")
            uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

            shape = [model.channels, h//8, w//8]
            samples_cfg, intermediates = sampler.sample(
                    ddim_steps,
                    num_samples,
                    shape,
                    cond,
                    verbose=False,
                    eta=1.0,
                    unconditional_guidance_scale=scale,
                    unconditional_conditioning=uc_full,
                    x_T=start_code,
            )
            x_samples_ddim = model.decode_first_stage(samples_cfg)

            result = torch.clamp((x_samples_ddim+1.0)/2.0,
                                 min=0.0, max=1.0)

            result = result.cpu().numpy().transpose(0,2,3,1)
            result, has_nsfw_concept = check_safety(result)
            result = result*255

    result = [Image.fromarray(img.astype(np.uint8)) for img in result]
    result = [put_watermark(img) for img in result]
    return result


def run():
    st.title("Stable Diffusion Inpainting")
    
    sampler = initialize_model(sys.argv[1], sys.argv[2])

    image = st.file_uploader("Image", ["jpg", "png"])
    if image:
        image = Image.open(image)
        w, h = image.size
        print(f"loaded input image of size ({w}, {h})")
        if max(w, h) > MAX_SIZE:
            factor = MAX_SIZE / max(w, h)
            w = int(factor*w)
            h = int(factor*h)
        width, height = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
        image = image.resize((width, height))
        print(f"resized to ({width}, {height})")

        prompt = st.text_input("Prompt")

        seed = st.number_input("Seed", min_value=0, max_value=1000000, value=0)
        num_samples = st.number_input("Number of Samples", min_value=1, max_value=64, value=1)
        scale = st.slider("Scale", min_value=0.1, max_value=30.0, value=7.5, step=0.1)
        ddim_steps = st.slider("DDIM Steps", min_value=0, max_value=50, value=50, step=1)

        fill_color = "rgba(255, 255, 255, 0.0)"
        stroke_width = st.number_input("Brush Size",
                                       value=64,
                                       min_value=1,
                                       max_value=100)
        stroke_color = "rgba(255, 255, 255, 1.0)"
        bg_color = "rgba(0, 0, 0, 1.0)"
        drawing_mode = "freedraw"

        st.write("Canvas")
        st.caption("Draw a mask to inpaint, then click the 'Send to Streamlit' button (bottom left, with an arrow on it).")
        canvas_result = st_canvas(
            fill_color=fill_color,
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            background_image=image,
            update_streamlit=False,
            height=height,
            width=width,
            drawing_mode=drawing_mode,
            key="canvas",
        )
        if canvas_result:
            mask = canvas_result.image_data
            mask = mask[:, :, -1] > 0
            if mask.sum() > 0:
                mask = Image.fromarray(mask)

                result = inpaint(
                    sampler=sampler,
                    image=image,
                    mask=mask,
                    prompt=prompt,
                    seed=seed,
                    scale=scale,
                    ddim_steps=ddim_steps,
                    num_samples=num_samples,
                    h=height, w=width
                )
                st.write("Inpainted")
                for image in result:
                    st.image(image)


def mask_process(mask):
    
    kernel_size = 7
    fire_thred = 0.9
    # zone_overlap_thred = 0.9
    # flame_thred = 0.1
    # num_areas = 100
    
    # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = mask
    # w,h = gray.shape
    # ax2 = plt.subplot(2,2,2)
    # ax2.imshow(gray)
    # ax1.colorbar()
    # gray = cv2.blur(gray, (kernel_size,kernel_size))
    # image_IR =  cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # print(np.mean(gray))
    # print(np.median(gray))
    # print(np.max(gray))
    
    thred = np.mean(gray) + np.median(gray)

    # np.sort(gray.reshape(-1))
    gray_sort = np.sort(gray.reshape(-1))
    
    fire_line = gray_sort[int(len(gray_sort)*fire_thred)]

    # X2 = gray_sort
    # F2 = np.array(range(len(X2)))/float(len(X2))
    # plt.plot(X2, F2)
    
    # plt.scatter(np.arange(len(gray.reshape(-1))),np.sort(gray.reshape(-1)))
    seg = copy.deepcopy(gray)
    seg[seg < fire_line] = 0
    # plt.imshow(seg)
    seg_blur = cv2.blur(seg, (kernel_size,kernel_size))
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))

    mask = cv2.morphologyEx(seg_blur, cv2.MORPH_OPEN, kernel, iterations = 3)
    mask = cv2.morphologyEx(seg_blur, cv2.MORPH_CLOSE, kernel, iterations = 3)


    # plt.imshow(seg_blur)
    # mask =  cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    # mask =  cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    # plt.imshow(mask)
    return mask


import numpy as np
import random

def generate_random_mask_bbox(image_height, image_width):
    # Generate a random shape for the mask
    mask_shape = random.choice(['circle', 'rectangle'])

    # Create an empty mask
    mask = np.zeros((image_height, image_width), dtype=bool)
    x, y, w, h = 0, 0, 0, 0  # Initialize bbox coordinates

    # Generate mask and bounding box based on the selected shape
    if mask_shape == 'circle':
        center_x = random.randint(0, image_width - 1)
        center_y = random.randint(0, image_height - 1)
        radius = random.randint(200, min(image_height, image_width) // 2)

        y, x = np.ogrid[:image_height, :image_width]
        mask[((x - center_x)**2 + (y - center_y)**2) <= radius**2] = True
        x, y, w, h = center_x - radius, center_y - radius, 2 * radius, 2 * radius

    elif mask_shape == 'rectangle':
        w = random.randint(200, image_width//2)
        h = random.randint(200, image_width//2)
        start_x = random.randint(0, image_width - 1 - w)
        start_y = random.randint(0, image_height - 1 - h)
        # end_x = random.randint(start_x, image_width - 1)
        # end_y = random.randint(start_y, image_height - 1)

        mask[start_y:start_y+h, start_x:start_x+h] = True
        # x, y, w, h = start_x, start_y, end_x - start_x, end_y - start_y
        x, y, w, h = start_x, start_y, w, h 

    
    return mask, [x, y, w, h]
    
    
import PIL
import os
import copy
import random

if __name__ == "__main__":
    # run()
    
    config = './configs/stable-diffusion/v1-inpainting-inference.yaml'
    ckpt   = './models/ldm/inpainting_big/sd-v1-5-inpainting.ckpt'
    
    
    img_path = 'D:/Data/Flame 2/254p Dataset/254p RGB Images/'
    # mask_path = 'D:/Data/Flame 2/254p Dataset/254p Thermal Images/'
    img_list = os.listdir(img_path)
    
    img_file = random.choice(img_list)
    image_name = img_file.split('.')[0]
    
    image = './bus.jpg'#img_path+img_file
    image_size = 512
    
    prompt = "fire"
    seed = random.randint(0, 1000000)
    num_samples = 1
    scale = 10
    # random.uniform(5, 15)
    ddim_steps = 10
    # st.slider("DDIM Steps", min_value=0, max_value=50, value=50, step=1)


    
    
    st.title("Stable Diffusion Inpainting")
    
    sampler = initialize_model(config, ckpt)

    # image = st.file_uploader("Image", ["jpg", "png"])
    if image:
        image = Image.open(image)
        # mask = Image.open(mask_path+img_file).convert('L')
        w, h = (image_size,image_size)
        print(f"loaded input image of size ({w}, {h})")
        if max(w, h) > MAX_SIZE:
            factor = MAX_SIZE / max(w, h)
            w = int(factor*w)
            h = int(factor*h)
        width, height = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
        image = image.resize((width, height))
        # mask  = mask.resize((width, height))
        print(f"resized to ({width}, {height})")
        
        
        mask, bbox = generate_random_mask_bbox(image_size,image_size)
        mask = np.array(mask)
        # mask = mask_process(mask)
        mask = mask.astype(bool)
        mask = Image.fromarray(mask)

        
        
        
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        model = sampler.model

        prng = np.random.RandomState(seed)
        start_code = prng.randn(num_samples, 4, h//8, w//8)
        start_code = torch.from_numpy(start_code).to(device=device, dtype=torch.float32)

        with torch.no_grad():
            with torch.autocast("cuda"):
                batch = make_batch_sd(image, mask, txt=prompt, device=device, num_samples=num_samples)

                c = model.cond_stage_model.encode(batch["txt"])

                c_cat = list()
                for ck in model.concat_keys:
                    cc = batch[ck].float()
                    if ck != model.masked_image_key:
                        bchw = [num_samples, 4, h//8, w//8]
                        cc = torch.nn.functional.interpolate(cc, size=bchw[-2:])
                    else:
                        cc = model.get_first_stage_encoding(model.encode_first_stage(cc))
                    c_cat.append(cc)
                c_cat = torch.cat(c_cat, dim=1)

                # cond
                cond={"c_concat": [c_cat], "c_crossattn": [c]}

                # uncond cond
                uc_cross = model.get_unconditional_conditioning(num_samples, "")
                uc_full = {"c_concat": [c_cat], "c_crossattn": [uc_cross]}

                shape = [model.channels, h//8, w//8]
                samples_cfg, intermediates = sampler.sample(
                        ddim_steps,
                        num_samples,
                        shape,
                        cond,
                        verbose=False,
                        eta=1.0,
                        unconditional_guidance_scale=scale,
                        unconditional_conditioning=uc_full,
                        x_T=start_code,
                )
                x_samples_ddim = model.decode_first_stage(samples_cfg)

                result = torch.clamp((x_samples_ddim+1.0)/2.0,
                                     min=0.0, max=1.0)

                result = result.cpu().numpy().transpose(0,2,3,1)
                result, has_nsfw_concept = check_safety(result)
                result = result*255

        result = [Image.fromarray(img.astype(np.uint8)) for img in result]
        
        output = np.array(result[0])
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        
        mask = np.array(mask).astype(np.uint8) *255
        cv2.imwrite('inpaint.png', output)
        cv2.imwrite('mask.png', mask)

        # result = [put_watermark(img) for img in result]
        
        
    
        # return result

        # prompt = st.text_input("Prompt")

        # seed = st.number_input("Seed", min_value=0, max_value=1000000, value=0)
        # num_samples = st.number_input("Number of Samples", min_value=1, max_value=64, value=1)
        # scale = st.slider("Scale", min_value=0.1, max_value=30.0, value=7.5, step=0.1)
        # ddim_steps = st.slider("DDIM Steps", min_value=0, max_value=50, value=50, step=1)

        # fill_color = "rgba(255, 255, 255, 0.0)"
        # stroke_width = st.number_input("Brush Size",
        #                                value=64,
        #                                min_value=1,
        #                                max_value=100)
        # stroke_color = "rgba(255, 255, 255, 1.0)"
        # bg_color = "rgba(0, 0, 0, 1.0)"
        # drawing_mode = "freedraw"

        # st.write("Canvas")
        # st.caption("Draw a mask to inpaint, then click the 'Send to Streamlit' button (bottom left, with an arrow on it).")
        # canvas_result = st_canvas(
        #     fill_color=fill_color,
        #     stroke_width=stroke_width,
        #     stroke_color=stroke_color,
        #     background_color=bg_color,
        #     background_image=image,
        #     update_streamlit=False,
        #     height=height,
        #     width=width,
        #     drawing_mode=drawing_mode,
        #     key="canvas",
        # )
        
        
        
        
        
        
        
        
        # if canvas_result:
        #     mask = canvas_result.image_data
        #     mask = mask[:, :, -1] > 0
        #     if mask.sum() > 0:
        #         mask = Image.fromarray(mask)

        #         result = inpaint(
        #             sampler=sampler,
        #             image=image,
        #             mask=mask,
        #             prompt=prompt,
        #             seed=seed,
        #             scale=scale,
        #             ddim_steps=ddim_steps,
        #             num_samples=num_samples,
        #             h=height, w=width
        #         )
        #         st.write("Inpainted")
        #         for image in result:
        #             st.image(image)

