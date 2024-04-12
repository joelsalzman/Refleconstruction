import numpy as np

def reflect_points(points : np.array, plane_normal : np.array) -> np.array:
    """
    Inputs are the point cloud and the plane they'll be reflected over.
    Output is a point cloud.
    """

    # Normalize the plane normal
    plane_normal = np.array(plane_normal)
    plane_normal = plane_normal / np.linalg.norm(plane_normal)

    # Convert points to a numpy array
    points = np.array(points)

    # Reflect points
    reflected_points = points - 2 * np.dot(points, plane_normal)[:, np.newaxis] * plane_normal

    return reflected_points

def find_correspondence(image, mask1, mask2):
    pass

def regularize_image(img, target_img, correspondences):
    pass