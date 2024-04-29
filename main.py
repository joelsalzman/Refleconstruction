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
from SAM import SAM_input, mask_to_points
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

    color = np.asanyarray(color_frame.get_data())
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    images = np.hstack((color, colorized_depth))
    plt.imshow(images)
    plt.show()
    
    obj_mask, mirr_mask, ref_mask = SAM_input(img, depth_frame, profile, output=True)
    
    # GET DEPTH DATA AT EACH POINT
    """
    1. overlay the mask on the depth map
    2. at each pixel where the depth map is non 0 copy rgb d to new img array
    3. save
    """
    # depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    
    # mirror_point_cloud = mask_to_points(mirr_mask, depth_frame, 
    #                                     depth_profile.get_intrinsics())
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(mirror_point_cloud)
    # o3d.io.write_point_cloud('mirror_point_cloud.ply', pcd)
    
    # object_point_cloud = mask_to_points(obj_mask, depth_frame, 
    #                                     depth_profile.get_intrinsics())
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(object_point_cloud)
    # o3d.io.write_point_cloud('object_point_cloud.ply', pcd)
    
    # ref_point_cloud = mask_to_points(ref_mask, depth_frame, 
    #                                  depth_profile.get_intrinsics())
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(ref_point_cloud)
    
    direct_mask = load_rgb(r'data\segmented\direct_mask.png'
                           ).any(axis=2).astype('uint8')
    reflect_mask = load_rgb(r'data\segmented\reflect_mask.png'
                           ).any(axis=2).astype('uint8')
    
    img, depth, intrinsics = from_realsense(filepath)

    match_points = SIFT(img, direct_mask, reflect_mask, threshold=.5, seams=False)

    normal = compute_normal(match_points, depth, intrinsics)

    run_model(normal)