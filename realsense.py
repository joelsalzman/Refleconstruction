# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 18:39:47 2024

@author: JOEL
"""

import numpy as np
import pyrealsense2 as rs  
import cv2

def from_realsense(filepath):
    
    # Setup:
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(filepath)
    profile = pipe.start(cfg)
    
    # Skip 5 first frames to give the Auto-Exposure time to adjust
    for x in range(5):
        pipe.wait_for_frames()
      
    # Store next frameset for later processing:
    frameset = pipe.wait_for_frames()
    color_frame = frameset.get_color_frame()
    
    # Cleanup:
    pipe.stop()
    
    align = rs.align(rs.stream.color)
    frameset = align.process(frameset)    
    aligned_depth_frame = frameset.get_depth_frame()
    
    color = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)
    depth_raw = np.asanyarray(aligned_depth_frame.get_data())
    
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
    depth = depth_raw.astype(np.float64) * depth_scale
    
    ds_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
    intrinsics = ds_profile.get_intrinsics()
    
    return color, depth, intrinsics
    

def capture_frames(filepath):
    """captures and returns color and depth frames"""
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(filepath)
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
    
    color, depth, intrinsics = from_realsense(filepath=r"data\bags\b1.bag")
    
