import open3d as o3d
import cv2  # state of the art computer vision algorithms library
import numpy as np  # fundamental package for scientific computing
import matplotlib.pyplot as plt  # 2D plotting library producing publication quality figures
import pyrealsense2 as rs  # Intel RealSense cross-platform open-source API
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
parser.add_argument("-d", type=bool, help="demo")

args = parser.parse_args()
test = args.t
demo = args.d

# LOAD IN MODEL
# device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
device = torch.device("cuda:1")

# if test:
model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-base")


# CODE
def mask_to_points(mask, vertices, tex_coords):

    filtered_indices = np.where(mask.flatten())[
        0
    ]  # Ensure mirr_mask is flattened and the same shape as vertices
    filtered_vertices = vertices[filtered_indices]
    filtered_tex_coords = tex_coords[filtered_indices]

    # Filter out vertices with zero depth
    valid_depth_indices = filtered_vertices[:, 2] != 0
    filtered_vertices = filtered_vertices[valid_depth_indices]
    filtered_tex_coords = filtered_tex_coords[valid_depth_indices]

    # Extract colors using the filtered and valid texture coordinates
    colors = np.zeros((len(filtered_tex_coords), 3), dtype=np.uint8)
    for i, tex_coord in enumerate(filtered_tex_coords):
        u, v = int(tex_coord[0] * color_image.shape[1]), int(
            tex_coord[1] * color_image.shape[0]
        )
        if 0 <= u < color_image.shape[1] and 0 <= v < color_image.shape[0]:
            colors[i] = color_image[v, u]  # Ensure indexing is within bounds

    # Create and visualize the point cloud
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(filtered_vertices)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors

    return pcd


# TODO move everything into the pipeline and do a big refactor
def setup_pipeline():
    """captures and returns color and depth frames"""
    pipe = rs.pipeline()
    cfg = rs.config()

    if demo:
        cfg.enable_device_from_file("data/bag1.bag")
        print("Configured for BAG file playback.")
    else:
        cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        print("Configured for live data stream from camera.")

    profile = pipe.start(cfg)

    for x in range(10):
        pipe.wait_for_frames()

    frameset = pipe.wait_for_frames()

    align = rs.align(rs.stream.color)
    frameset = align.process(frameset)

    color_frame = frameset.get_color_frame()
    depth_frame = frameset.get_depth_frame()

    if not color_frame or not depth_frame:
        raise ValueError("Could not obtain color or depth frames.")

    return color_frame, depth_frame, pipe, profile


if __name__ == "__main__":
    # filepath_color = "./data/parrot_test_5_Color.png"
    # segment_and_depth(filepath_color)

    # color_frame, depth_frame, pipe, profile = capture_frames()

    # TODO add a flag to either read from bag file or to do on the fly

    # pipe = rs.pipeline()
    # cfg = rs.config()
    # cfg.enable_device_from_file("data/bag1.bag")
    # profile = pipe.start(cfg)

    # # Skip 5 first frames to give the Auto-Exposure time to adjust
    # for x in range(10):
    #     pipe.wait_for_frames()

    # # Store next frameset for later processing:
    # frameset = pipe.wait_for_frames()

    # align = rs.align(rs.stream.color)
    # frameset = align.process(frameset)

    # color_frame = frameset.get_color_frame()
    # depth_frame = frameset.get_depth_frame()

    color_frame, depth_frame, pipe, profile = setup_pipeline(demo)

    colorizer = rs.colorizer()

    color_image = np.asanyarray(color_frame.get_data())
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    images = np.hstack((color_image, colorized_depth))
    plt.imshow(images)
    # plt.show()

    # Segment oout the points or use known coords
    if test:
        point_obj = (448.07, 293.53)
        point_mirr = (80.54, 220.80)
        point_ref = (123.40, 250.67)

    else:
        print("### SELECT OBJECT ###")
        point_obj = get_point_from_image(color_image, "### SELECT OBJECT ###")
        print("### SELECT MIRROR ###")
        point_mirr = get_point_from_image(color_image, "### SELECT MIRROR ###")
        print("### SELECT REFLECTION ###")
        point_ref = get_point_from_image(color_image, "### SELECT REFLECTION ###")

    print("Segmenting objects...")
    obj_mask = segment_with_sam(
        model, processor, [[[point_obj[0], point_obj[1]]]], color_image
    )[0][0][0].numpy()
    print("Segmenting objects...")
    mirr_mask = segment_with_sam(
        model, processor, [[[point_mirr[0], point_mirr[1]]]], color_image
    )[0][0][
        2
    ].numpy()  # keep mask 2
    print("Segmenting objects...")
    ref_mask = segment_with_sam(
        model, processor, [[[point_ref[0], point_ref[1]]]], color_image
    )[0][0][0].numpy()
    print("Done segmenting...")

    ### After segmenting get the depthitre
    height, width = color_image.shape[:2]
    expected = 300
    aspect = width / height

    depth = np.asanyarray(depth_frame.get_data())
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    obj_depth = depth[obj_mask > 0].astype(float) * depth_scale
    mirror_depth = depth[mirr_mask > 0].astype(float) * depth_scale
    ref_depth = depth[ref_mask > 0].astype(float) * depth_scale
    # depth = depth * depth_scale
    mdist, _, _, _ = cv2.mean(obj_depth)
    odist, _, _, _ = cv2.mean(mirror_depth)
    rdist, _, _, _ = cv2.mean(ref_depth)

    print(f"Detected obj {mdist:.3} meters away.")
    print(f"Detected mirror {odist:.3} meters away.")
    print(f"Detected reflection {rdist:.3} meters away.")

    messages = [
        f"Detected obj {mdist:.3} meters away.",
        f"Detected mirror {odist:.3} meters away.",
        f"Detected reflection {rdist:.3} meters away.",
    ]

    # VISUALIZE MAPS
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    axes[0].imshow(color_image)
    axes[0].set_title("Original Image")
    mask_list = [obj_mask, mirr_mask, ref_mask]

    for i, mask in enumerate(mask_list, start=1):
        overlayed_image = np.array(color_image).copy()

        overlayed_image[:, :, 0] = np.where(mask == 1, 255, overlayed_image[:, :, 0])
        # overlayed_image[:,:,1] = np.where(mask == 1, 0, overlayed_image[:,:,1])
        # overlayed_image[:,:,2] = np.where(mask == 1, 0, overlayed_image[:,:,2])

        axes[i].imshow(overlayed_image)
        axes[i].set_title(messages[i - 1])
    for ax in axes:
        ax.axis("off")
    plt.show()

    # GET DEPTH DATA AT EACH POINT
    """
    1. overlay the mask on the depth map
    2. at each pixel where the depth map is non 0 copy rgb d to new img array
    3. save
    """

    # use clusterint to get rid of background noise
    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))

    intrinsics = depth_profile.get_intrinsics()

    depth_image = np.asanyarray(depth_frame.get_data())

    pc = rs.pointcloud()
    pc.map_to(color_frame)
    points = pc.calculate(depth_frame)

    vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    tex_coords = (
        np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
    )

    mirror_pcd = mask_to_points(mirr_mask, vertices, tex_coords)
    obj_pcd = mask_to_points(obj_mask, vertices, tex_coords)
    ref_pcd = mask_to_points(ref_mask, vertices, tex_coords)

    print("saving point clouds individually")

    o3d.io.write_point_cloud("mirror_point_colour_cloud.ply", mirror_pcd)
    o3d.io.write_point_cloud("obj_point_colour_cloud.ply", obj_pcd)
    o3d.io.write_point_cloud("ref_point_colour_cloud.ply", ref_pcd)

    o3d.visualization.draw_geometries(
        [mirror_pcd, obj_pcd, ref_pcd],
        window_name="segmented color mapped point clouds",
        width=800,
        height=800,
        left=50,
        top=50,
    )

    # Do you need this? idk
    pipe.stop()

    # mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)[0]

    # o3d.io.write_triangle_mesh("outtest.ply", mesh, write_vertex_colors=True)

    # mirror_point_cloud = mask_to_mesh(
    #     mirr_mask, depth_frame, depth_profile.get_intrinsics()
    # )
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(mirror_point_cloud)
    # o3d.io.write_point_cloud("mirror_point_mesh.ply", pcd)

    # object_point_cloud = mask_to_mesh(
    #     obj_mask, depth_frame, depth_profile.get_intrinsics()
    # )
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(object_point_cloud)
    # o3d.io.write_point_cloud("object_point_mesh.ply", pcd)

    # ref_point_cloud = mask_to_mesh(
    #     ref_mask, depth_frame, depth_profile.get_intrinsics()
    # )
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(ref_point_cloud)
    # o3d.io.write_point_cloud("ref_point_mesh.ply", pcd)

    #

    # filtered_indices = np.where(mirr_mask.flatten())[0]  # Ensure mirr_mask is flattened and the same shape as vertices
    # filtered_vertices = vertices[filtered_indices]
    # filtered_tex_coords = tex_coords[filtered_indices]

    # # Filter out vertices with zero depth
    # valid_depth_indices = filtered_vertices[:, 2] != 0
    # filtered_vertices = filtered_vertices[valid_depth_indices]
    # filtered_tex_coords = filtered_tex_coords[valid_depth_indices]

    # # Extract colors using the filtered and valid texture coordinates
    # colors = np.zeros((len(filtered_tex_coords), 3), dtype=np.uint8)
    # for i, tex_coord in enumerate(filtered_tex_coords):
    #     u, v = int(tex_coord[0] * color_image.shape[1]), int(tex_coord[1] * color_image.shape[0])
    #     if 0 <= u < color_image.shape[1] and 0 <= v < color_image.shape[0]:
    #         colors[i] = color_image[v, u]  # Ensure indexing is within bounds

    # print(np.unique(colors, axis=0))

    # # Create and visualize the point cloud
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(filtered_vertices)
    # pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # Normalize colors
