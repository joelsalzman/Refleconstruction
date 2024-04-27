import numpy as np
import bpy
import bmesh
import mathutils
from mathutils import Vector, Matrix
from sklearn.neighbors import NearestNeighbors

def reflect_points(points, plane_normal):
    """
    Inputs are the point cloud and the plane they'll be reflected over.
    Output is a point cloud.
    """
    # Normalize the plane normal
    plane_normal.normalize()
    
    # Convert points to a list of mathutils.Vector
    points = [mathutils.Vector(point) for point in points]
    
    # Reflect points
    reflected_points = []
    for point in points:
        projection = point.dot(plane_normal)
        reflected_point = point - 2 * projection * plane_normal
        reflected_points.append(reflected_point)
    
    return reflected_points

def find_plane_candidates(bm):
    
    obj = bpy.context.active_object
    
    # Generate the convex hull
    bmesh.ops.convex_hull(bm, input=bm.verts)
    
    # Update the mesh with the bmesh changes
    bm.to_mesh(obj.data)
    bm.free()
    
    # Switch back to object mode
    bpy.ops.object.mode_set(mode='OBJECT')
    
    # Output the plane normals for every face in the convex hull
    mesh = obj.data
    normals = [poly.normal for poly in mesh.polygons]
    
    return normals

def _flip_merge(r1, r2, direct, r1_n, r2_n, i):

    f1 = reflect_points(r1, r1_n)
    f2 = reflect_points(r2, r2_n)

    merged_mesh = bpy.data.meshes.new(name=f"Candidate {i}")
    merged_obj = bpy.data.objects.new(f"Candidate {i}", merged_mesh)
    bpy.context.scene.collection.objects.link(merged_obj)  
    bm = bmesh.new()

    for obj in [f1, f2, direct]:
        if obj.type == 'MESH':
            mesh = obj.data
            for vert in mesh.vertices:
                bm.verts.new(vert.co)

    # Update the merged mesh with the bmesh
    bm.to_mesh(merged_mesh)
    bm.free()

    # Update the scene
    bpy.context.view_layer.update()

    return merged_mesh

def _gaussian_curvature(points, k=10):

    # Create a nearest neighbor object
    nbrs = NearestNeighbors(n_neighbors=k+1, algorithm='ball_tree').fit(points)
    
    # Find the indices and distances of the k-nearest neighbors for each point
    _, indices = nbrs.kneighbors(points)
    
    # Initialize a list to store the Gaussian curvature values
    gaussian_curvatures = [0.0] * len(points)
    
    # Iterate over each point
    for i in range(len(points)):
        # Get the neighboring points
        neighbors = [Vector(points[j]) for j in indices[i][1:]]
        
        # Center the neighboring points
        centered_neighbors = [p - Vector(points[i]) for p in neighbors]
        
        # Compute the covariance matrix of the centered neighbors
        cov_matrix = Matrix.Covariance(centered_neighbors)
        
        # Compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues = cov_matrix.eigenvalues()
        
        # Compute the Gaussian curvature
        gaussian_curvatures[i] = (eigenvalues[0] * eigenvalues[1]) / (sum(eigenvalues) ** 2)
    
    return gaussian_curvatures


def find_reflection(direct, r1, r2):

    r1_cand = find_plane_candidates(r1)
    r2_cand = find_plane_candidates(r2)

    # How to possibly flip the two reflected point clouds
    flips = [(r1_n, r2_n) for r1_n in r1_cand for r2_n in r2_cand]
    for i in range(len(flips)):
        r1_n, r2_n = flips[i]
        mesh = _flip_merge(r1, r2, direct, r1_n, r2_n, i)

