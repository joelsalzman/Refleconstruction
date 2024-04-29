import numpy as np
# import bpy
# import bmesh
# import mathutils
# from mathutils import Vector, Matrix
from sklearn.neighbors import NearestNeighbors
import open3d as o3d

# def reflect_points(points, plane_normal):
#     """
#     Inputs are the point cloud and the plane they'll be reflected over.
#     Output is a point cloud.
#     """
#     # Normalize the plane normal
#     plane_normal.normalize()
    
#     # Convert points to a list of mathutils.Vector
#     points = [mathutils.Vector(point) for point in points]
    
#     # Reflect points
#     reflected_points = []
#     for point in points:
#         projection = point.dot(plane_normal)
#         reflected_point = point - 2 * projection * plane_normal
#         reflected_points.append(reflected_point)
    
#     return reflected_points

# def find_plane_candidates(bm):
    
#     obj = bpy.context.active_object
    
#     # Generate the convex hull
#     bmesh.ops.convex_hull(bm, input=bm.verts)
    
#     # Update the mesh with the bmesh changes
#     bm.to_mesh(obj.data)
#     bm.free()
    
#     # Switch back to object mode
#     bpy.ops.object.mode_set(mode='OBJECT')
    
#     # Output the plane normals for every face in the convex hull
#     mesh = obj.data
#     normals = [poly.normal for poly in mesh.polygons]
    
#     return normals

# def _flip_merge(r1, r2, direct, r1_n, r2_n, i):

#     f1 = reflect_points(r1, r1_n)
#     f2 = reflect_points(r2, r2_n)

#     merged_mesh = bpy.data.meshes.new(name=f"Candidate {i}")
#     merged_obj = bpy.data.objects.new(f"Candidate {i}", merged_mesh)
#     bpy.context.scene.collection.objects.link(merged_obj)  
#     bm = bmesh.new()

#     for obj in [f1, f2, direct]:
#         if obj.type == 'MESH':
#             mesh = obj.data
#             for vert in mesh.vertices:
#                 bm.verts.new(vert.co)

#     # Update the merged mesh with the bmesh
#     bm.to_mesh(merged_mesh)
#     bm.free()

#     # Update the scene
#     bpy.context.view_layer.update()

#     return merged_mesh

def mask_to_points(mask, vertices, tex_coords, color_image, ):

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