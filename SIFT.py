import numpy as np
import cv2
import os
from scipy.ndimage import binary_dilation
from load import load_rgb
from utils import imshow, imshow_
from realsense import from_realsense
import pyrealsense2 as rs

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def estimate_seams(left, right):
    
    # Create seams
    left_buffer = binary_dilation(left.copy(), iterations=2).astype(bool)
    right_buffer = binary_dilation(right.copy(), iterations=3).astype(bool)
    seams_left = left_buffer ^ ~binary_dilation(~left_buffer, iterations=9)
    seams_right = right_buffer ^ ~binary_dilation(~right_buffer, iterations=15)
    
    return seams_left, seams_right


def SIFT(img, direct_mask, reflect_mask, threshold=.5, seams=False):

    # Based on the setup, we can guess what is supposed to align with what
    left, right = None, None
    for col in range(img.shape[1]):
        if direct_mask[:, col].any():
            left, right = direct_mask, reflect_mask
            break
        elif reflect_mask[:, col].any():
            left, right = reflect_mask, direct_mask
            break
        
    if seams:
        left, right = estimate_seams(left, right)

    # Detect SIFT features
    sift = cv2.SIFT_create()
    kp_l, des_l = sift.detectAndCompute(img, left.astype('uint8'))
    kp_r, des_r = sift.detectAndCompute(img, right.astype('uint8'))

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_l, des_r, k=2)

    # Apply ratio test
    good_matches = [[m] for m, n in matches if m.distance < threshold*n.distance]
    while not good_matches:
        threshold += .2
        good_matches = [[m] for m, n in matches if m.distance < threshold*n.distance]
        if not good_matches and threshold >= 1:
            raise Exception('Failed to find reasonable SIFT matches')

    # Sanity check
    print(f'Found {len(good_matches)} matches from {len(kp_l)} : {len(kp_r)}')
    match_img = cv2.drawMatchesKnn(
        img * np.stack((left,)*3, axis=2),
        kp_l,
        img * np.stack((right,)*3, axis=2),
        kp_r,
        good_matches,
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    imshow_(match_img)

    match_points = np.array([
        (kp_l[match[0].queryIdx].pt, kp_r[match[0].trainIdx].pt)
        for match in good_matches
    ])

    return match_points


def compute_xyz_coordinates(depth, intrinsics):
    
    fx = intrinsics.fx
    fy = intrinsics.fy
    cx = intrinsics.ppx
    cy = intrinsics.ppy
    
    height, width = depth.shape
    
    # Create meshgrid of pixel coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    print(x.shape, y.shape)
    
    # Normalize pixel coordinates
    x = (x - cx) / fx
    y = (y - cy) / fy
    
    # Compute 3D coordinates
    X = x * depth
    Y = y * depth
    Z = depth
    
    # Stack coordinates into a 3-channel image
    xyz = np.stack((X, Y, Z), axis=2)
    
    return xyz

def compute_normal(match_points, depth, intrinsics):

    pixels = match_points.astype(int)
    indices = np.array([pixels[:, i, :][0] for i in range(pixels.shape[1])])

    xyz = compute_xyz_coordinates(depth, intrinsics)
    coords = xyz[indices[:, 1], indices[:, 0], :]

    sixdof = list()
    for i in range(0, coords.shape[0], 2):
        location = (coords[i, :] + coords[i+1, :]) / 2
        orientation = coords[i, :] - coords[i+1, :]
        orientation /= np.linalg.norm(orientation)
        sixdof.append([*location, *orientation])
        
    final = np.array(sixdof).mean(axis=0)
    
    return final


if __name__ == '__main__':

    filepath = r"data\bags\objectsbook.bag"
    
    direct_mask = load_rgb(r'data\segmented\direct_mask.png'
                           ).any(axis=2).astype('uint8')
    reflect_mask = load_rgb(r'data\segmented\reflect_mask.png'
                           ).any(axis=2).astype('uint8')
    
    img, depth, intrinsics = from_realsense(filepath)

    match_points = SIFT(img, direct_mask, reflect_mask, seams=False)

    normal = compute_normal(match_points, depth, intrinsics)

    