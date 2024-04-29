import open3d as o3d 
import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API
import argparse

import torch
from transformers import SamModel, SamProcessor

# CUSTOMS
from load import load_ply
from segment import find_plane
from transform import reflect_points

from rgb_coords import get_point_from_image
from segment_rgb import *

# SET UP ARGS
parser = argparse.ArgumentParser(description="Process some integers.")
parser.add_argument("-t", type=bool, help="Test")
args = parser.parse_args()
test = args.t

# LOAD IN MODEL
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# if test:
model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")


 
 
# CODE
def mask_to_points(mask, depth_frame, depth_intr):
    points = []
    
    # print(depth_intr.height, depth_intr.width)
    # print(mask.shape)

    for y in range(depth_intr.height):
        for x in range(depth_intr.width):
            if mask[y,x]:
                if 0 <= x < width and 0 <= y < width:
                    depth = depth_frame.get_distance(x, y)
                    if depth > 0:
                        point = rs.rs2_deproject_pixel_to_point(depth_intr, [x,y], depth)
                        points.append(point)
    
    return np.array(points)

def capture_frames():
    """captures and returns color and depth frames"""
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file("data/bag1.bag")
    profile = pipe.start(cfg)

    # Skip 5 first frames to give the Auto-Exposure time to adjust
    for x in range(10):
        pipe.wait_for_frames()
    
    # Store next frameset for later processing:
    frameset = pipe.wait_for_frames()
    
    align = rs.align(rs.stream.color)
    frameset = align.process(frameset)
    
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()

    # Cleanup:
    pipe.stop()
    print("Frames Captured")
    return color_frame, depth_frame, pipe, profile
    
    
if __name__ == '__main__':
    # filepath_color = "./data/parrot_test_5_Color.png"
    # segment_and_depth(filepath_color)

    color_frame, depth_frame, pipe, profile = capture_frames()
    
    colorizer = rs.colorizer()

    color = np.asanyarray(color_frame.get_data())
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    images = np.hstack((color, colorized_depth))
    plt.imshow(images)
    plt.show()
    
    # Segment oout the points or use known coords
    if test:
        point_obj = (448.07, 293.53)
        point_mirr = (80.54, 220.80)
        point_ref = (123.40, 250.67)
        
    else:
        print("### SELECT OBJECT ###")
        point_obj = get_point_from_image(color, "### SELECT OBJECT ###")
        print("### SELECT MIRROR ###")
        point_mirr = get_point_from_image(color, "### SELECT MIRROR ###")
        print("### SELECT REFLECTION ###")
        point_ref = get_point_from_image(color, "### SELECT REFLECTION ###")
    
    print("Segmenting objects...")
    obj_mask = segment_with_sam(model, processor, [[[point_obj[0], point_obj[1]]]], color)[0][0][0].numpy()
    print("Segmenting objects...")
    mirr_mask = segment_with_sam(model, processor, [[[point_mirr[0], point_mirr[1]]]], color)[0][0][2].numpy() # keep mask 2
    print("Segmenting objects...")
    ref_mask = segment_with_sam(model, processor, [[[point_ref[0], point_ref[1]]]], color)[0][0][0].numpy()
    print("Done segmenting...")
    
    ### After segmenting get the depth 
    height, width = color.shape[:2]
    expected = 300
    aspect = width / height
    
    
    depth = np.asanyarray(depth_frame.get_data())
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    
    obj_depth = depth[obj_mask > 0].astype(float)* depth_scale
    mirror_depth = depth[mirr_mask > 0].astype(float)* depth_scale
    ref_depth = depth[ref_mask > 0].astype(float)* depth_scale
    # depth = depth * depth_scale
    mdist,_,_,_ = cv2.mean(obj_depth)
    odist,_,_,_ = cv2.mean(mirror_depth)
    rdist,_,_,_ = cv2.mean(ref_depth)
    
    
    print(f"Detected obj {mdist:.3} meters away.")
    print(f"Detected mirror {odist:.3} meters away.")
    print(f"Detected reflection {rdist:.3} meters away.")
    
    messages = [f"Detected obj {mdist:.3} meters away.", f"Detected mirror {odist:.3} meters away.", f"Detected reflection {rdist:.3} meters away."]
    
    
    # VISUALIZE MAPS
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    axes[0].imshow(color)
    axes[0].set_title('Original Image')
    mask_list = [obj_mask, mirr_mask, ref_mask]

    for i, mask in enumerate(mask_list, start=1):
        overlayed_image = np.array(color).copy()

        overlayed_image[:,:,0] = np.where(mask == 1, 255, overlayed_image[:,:,0])
        # overlayed_image[:,:,1] = np.where(mask == 1, 0, overlayed_image[:,:,1])
        # overlayed_image[:,:,2] = np.where(mask == 1, 0, overlayed_image[:,:,2])
        
        axes[i].imshow(overlayed_image)
        axes[i].set_title(messages[i - 1])
    for ax in axes:
        ax.axis('off')

    plt.show()
    
    # GET DEPTH DATA AT EACH POINT
    """
    1. overlay the mask on the depth map
    2. at each pixel where the depth map is non 0 copy rgb d to new img array
    3. save
    """
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    
    mirror_point_cloud = mask_to_points(mirr_mask, depth_frame, depth_profile.get_intrinsics())
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(mirror_point_cloud)
    o3d.io.write_point_cloud('mirror_point_cloud.ply', pcd)
    
    object_point_cloud = mask_to_points(obj_mask, depth_frame, depth_profile.get_intrinsics())
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(object_point_cloud)
    o3d.io.write_point_cloud('object_point_cloud.ply', pcd)
    
    ref_point_cloud = mask_to_points(ref_mask, depth_frame, depth_profile.get_intrinsics())
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ref_point_cloud)
    o3d.io.write_point_cloud('ref_point_cloud.ply', pcd)
    
    # 