import numpy as np
import cv2
import os
from scipy.ndimage import binary_dilation
from load import load_rgb
from utils import imshow, imshow_
import pyrealsense2 as rs

os.chdir(os.path.dirname(os.path.abspath(__file__)))


def estimate_seam(img, direct_mask, reflect_mask):

    # Based on the setup, we can guess what is supposed to align with what
    left, right = None, None
    for col in range(img.shape[1]):
        if direct_mask[:, col].any():
            left, right = direct_mask, reflect_mask
            break
        elif reflect_mask[:, col].any():
            left, right = reflect_mask, direct_mask
            break

    # Create seams
    left_buffer = binary_dilation(left.copy(), iterations=2).astype(bool)
    right_buffer = binary_dilation(right.copy(), iterations=3).astype(bool)
    seams_left = left_buffer ^ ~binary_dilation(~left_buffer, iterations=9)
    seams_right = right_buffer ^ ~binary_dilation(~right_buffer, iterations=15)

    # Detect SIFT features
    sift = cv2.SIFT_create()
    kp_l, des_l = sift.detectAndCompute(img, seams_left.astype('uint8'))
    kp_r, des_r = sift.detectAndCompute(img, seams_right.astype('uint8'))

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des_l, des_r, k=2)

    # Apply ratio test
    good_matches = [[m] for m, n in matches if m.distance < 0.7*n.distance]

    # Sanity check
    print(f'Found {len(good_matches)} matches from {len(kp_l)} : {len(kp_r)}')
    match_img = cv2.drawMatchesKnn(
        img * np.stack((seams_left,)*3, axis=2),
        kp_l,
        img * np.stack((seams_right,)*3, axis=2),
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


def compute_normal(match_points, depth):

    pixels = match_points.astype(int)

    indices = np.array([pixels[:, i, :][0] for i in range(pixels.shape[1])])
    depth_colors = depth[indices[:, 0], indices[:, 1], :]
    
    # TODO: get XYZ
    


if __name__ == '__main__':

    img = load_rgb(r'data\parrot_test_5_Color.png')

    depth = load_rgb(r'data\parrot_test_5_D_Depth.png')
    size = (img.shape[1], img.shape[0])
    depth = cv2.resize(depth, size, interpolation=cv2.INTER_LINEAR)

    direct_mask = load_rgb(r'data\masks\parrot_5_direct.png'
                           ).any(axis=2).astype('uint8')
    reflect_mask = load_rgb(r'data\masks\parrot_5_reflect.png'
                            ).any(axis=2).astype('uint8')

    match_points = estimate_seam(img, direct_mask, reflect_mask)

    compute_normal(match_points, depth)
