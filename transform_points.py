import numpy as np
import open3d as o3d
from pathlib import Path

def reflect_points(points : np.array, plane_normal : np.array) -> np.array:

    # Normalize the plane normal
    plane_normal = np.array(plane_normal)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # Convert points to a numpy array
    points = np.array(points)

    # Reflect points
    reflected_points = points - 2 * np.dot(points, plane_normal)[:, np.newaxis] * plane_normal

    return reflected_points

def find_plane(points):

    pass

def segment_reflected(points):

    pass

def load_ply(file_path):

    mesh = o3d.io.read_triangle_mesh("path/to/your/file.ply")
    vertices = np.asarray(mesh.vertices)
    return vertices

if __name__ == '__main__':

    point_cloud, faces = load_ply(r'data\parrot2.ply')

    plane_normal = find_plane(point_cloud)
    points = reflect_points(point_cloud, plane_normal)