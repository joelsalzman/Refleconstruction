import torch
from PIL import Image
import requests
from transformers import SamModel, SamProcessor
import matplotlib.pyplot as plt
import numpy as np
import cv2 

TEST_COORDS = [305, 127]
test_fp = '/home/research/Columbia/CI-Project/data/parrot_test_5_Color.png'

def resize_mask(mask, target_size):
    """ Resize the mask to the target size using OpenCV. """
    mask_uint8 = mask.astype(np.uint8) * 255  # Convert boolean to 0-255

    # Resize the mask using nearest interpolation
    resized_mask_uint8 = cv2.resize(mask_uint8, target_size[::-1], interpolation=cv2.INTER_NEAREST)

    # Convert back to boolean
    resized_mask = resized_mask_uint8 > 0  # Convert 255 back to True, 0 stays False

    return resized_mask

def upsample_depth(rgb_depth_map, target_size):
    """Upsamples the depth map to the dimension of the mask"""
    upsampled_rgb_depth_map = cv2.resize(rgb_depth_map, target_size[::-1], interpolation=cv2.INTER_CUBIC)
    return upsampled_rgb_depth_map

def segment_with_sam_fp(coords, filepath):
    """Loads and segments using sam"""
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")

    image = Image.open(filepath)
    
    inputs = processor(image, input_points=coords, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
    
    # we want to keep the first mask
    return masks[0][0][0].numpy

def segment_with_sam(model, processor, coords, image):
    """Loads and segments using sam"""
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    # processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    
    inputs = processor(image, input_points=coords, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model(**inputs)
    masks = processor.image_processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu())
    
    # we want to keep the first mask
    return masks

def apply_mask_to_map(mask, rgb):
    """Use the mask to retrieve the corresponding rgb"""
    # rgb depth map should have been loaded already
    # make sure that the map and te depth map have the same shape
    mask = mask.astype(bool)
    rgb_values = rgb[mask]
    return rgb_values

def shift_mask_right(mask, shift_x):

    if shift_x < 0:
        raise ValueError("shift_x must be a non-negative integer")
    
    shifted_mask = np.zeros_like(mask, dtype=bool)
    
    if shift_x > 0:
        shifted_mask[:, shift_x:] = mask[:, :-shift_x]
    
    return shifted_mask

def extract_mean_rgb(rgb_values):
    """Computes the mean in each channel with the nonzero elements"""
    masked_rgb = np.ma.masked_equal(rgb_values, 0)
    mean_rgb = np.ma.mean(masked_rgb, axis=0).filled(np.nan)  

    return mean_rgb
