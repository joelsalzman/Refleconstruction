import numpy as np
import open3d as o3d
import cv2

def load_ply(file_path):

    mesh = o3d.io.read_triangle_mesh("path/to/your/file.ply")
    vertices = np.asarray(mesh.vertices)
    return vertices

def load_rgb(file_path):

    return cv2.imread(file_path)