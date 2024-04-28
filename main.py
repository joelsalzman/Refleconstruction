import cv2                                # state of the art computer vision algorithms library
import numpy as np                        # fundamental package for scientific computing
import matplotlib.pyplot as plt           # 2D plotting library producing publication quality figures
import pyrealsense2 as rs                 # Intel RealSense cross-platform open-source API

from load import load_ply
from segment import find_plane
from transform import reflect_points

from rgb_coords import get_point_from_image

def segment_and_depth(filepath_color):
    get_point_from_image(filepath_color)
    

def capture_frames():
    """captures and returns color and depth frames"""
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file("../object_detection.bag")
    profile = pipe.start(cfg)

    # Skip 5 first frames to give the Auto-Exposure time to adjust
    for x in range(10):
        pipe.wait_for_frames()
    
    # Store next frameset for later processing:
    frameset = pipe.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()

    # Cleanup:
    pipe.stop()
    print("Frames Captured")
    return color_frame, depth_frame
    
    
if __name__ == '__main__':
    # filepath_color = "./data/parrot_test_5_Color.png"
    # segment_and_depth(filepath_color)

    color_frame, depth_frame = capture_frames()
    color = np.asanyarray(color_frame.get_data())
    
    # TODO using camera parameters map point and depth onto real world coords
