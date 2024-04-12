import numpy as np
import open3d as o3d

def load_ply(file_path):

    mesh = o3d.io.read_triangle_mesh("path/to/your/file.ply")
    vertices = np.asarray(mesh.vertices)
    return vertices

