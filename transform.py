import numpy as np
import open3d as o3d
import os
import cv2

def mask_to_points(mask, vertices, tex_coords, color_image):

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

def segment_point_clouds(
    basename, rs, profile, depth_frame, color_frame, obj_mask, mirr_mask, ref_mask):

    depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))

    intrinsics = depth_profile.get_intrinsics()

    depth_image = np.asanyarray(depth_frame.get_data())

    pc = rs.pointcloud()
    pc.map_to(color_frame)
    color_image = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)
    points = pc.calculate(depth_frame)

    vertices = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
    tex_coords = (
        np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
    )

    mirror_pcd = mask_to_points(mirr_mask, vertices, tex_coords, color_image)
    obj_pcd = mask_to_points(obj_mask, vertices, tex_coords, color_image)
    ref_pcd = mask_to_points(ref_mask, vertices, tex_coords, color_image)

    print("saving point clouds individually")

    o3d.io.write_point_cloud(os.path.join('data', 'segmented', f"{basename}_direct.ply"), obj_pcd)
    o3d.io.write_point_cloud(os.path.join('data', 'segmented', f"{basename}_mirror.ply"), mirror_pcd)
    o3d.io.write_point_cloud(os.path.join('data', 'segmented', f"{basename}_reflect.ply"), ref_pcd)

    o3d.visualization.draw_geometries(
        [mirror_pcd, obj_pcd, ref_pcd],
        window_name="segmented color mapped point clouds",
        width=800,
        height=800,
        left=50,
        top=50,
    )