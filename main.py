#@ -1,185 +0,0 @@
import open3d as o3d 
import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import argparse
import os

import torch
from transformers import SamModel, SamProcessor

# CUSTOMS
from rgb_coords import get_point_from_image
from segment_rgb import segment_with_sam
from realsense import from_realsense, capture_frames
from transform import mask_to_points, segment_point_clouds
from SAM import SAM_input
from load import load_rgb
from SIFT import SIFT, compute_normal
from run_model import run_model

# SET UP ARGS
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-t", type=bool, help="Test")
args = parser.parse_args()
test = args.t
 
# CODE
    
    
if __name__ == '__main__':
    # filepath_color = "./data/parrot_test_5_Color.png"
    # segment_and_depth(filepath_color)

    filepath = r"data\bags\trinkets.bag"
    name = os.path.basename(filepath)

    # TODO: these do very similar things
    color_frame, depth_frame, pipe, profile = capture_frames(filepath)
    img, depth, intrinsics = from_realsense(filepath)
    
    colorizer = rs.colorizer()

    color = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    images = np.hstack((color, colorized_depth))
    plt.imshow(images)
    plt.show()
    
    obj_mask, mirr_mask, ref_mask = SAM_input(
        filepath, rs, profile, color_frame, depth_frame, test=False, output=True)
    
    basename = os.path.basename(filepath).split('.')[0]
    dpath = os.path.join('data', 'segmented', f'{basename}_direct_mask.png')
    direct_mask = load_rgb(dpath).any(axis=2).astype('uint8')
    rpath = os.path.join('data', 'segmented', f'{basename}_reflect_mask.png')
    reflect_mask = load_rgb(rpath).any(axis=2).astype('uint8')
    
    img, depth, intrinsics = from_realsense(filepath)

    match_points = SIFT(img, direct_mask, reflect_mask, threshold=.5, seams=False)

    normal = compute_normal(match_points, depth, intrinsics)

    segment_point_clouds(basename, rs, profile, depth_frame, color_frame, obj_mask, mirr_mask, ref_mask)

    run_model(normal)