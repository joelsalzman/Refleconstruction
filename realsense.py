# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 18:39:47 2024

@author: JOEL
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import pyrealsense2 as rs  


def start():
    
    # Setup:
    pipe = rs.pipeline()
    cfg = rs.config()
    cfg.enable_device_from_file(r"data\bags\b1.bag")
    profile = pipe.start(cfg)
    
    # Skip 5 first frames to give the Auto-Exposure time to adjust
    for x in range(5):
      pipe.wait_for_frames()
      
    # Store next frameset for later processing:
    frameset = pipe.wait_for_frames()
    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()
    
    # Cleanup:
    pipe.stop()
    print("Frames Captured")
