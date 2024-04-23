import numpy as np

def find_plane(points):
    """
    Fits a plane to a point cloud. 
    Used for finding the equation of a flat mirror.
    """

    # Center the point cloud
    centroid = np.mean(points, axis=0)
    centered_points = points - centroid
    
    # Compute the covariance matrix
    cov_matrix = np.cov(centered_points.T)
    
    # Find the smallest eigenvector
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)
    plane_normal = eigenvectors[:, np.argmin(eigenvalues)]

    return plane_normal

def point_segmentation(points, mask) -> np.array:
    """
    Segments points based on a mask.
    """

    # TODO: Nikolaus
    pass 

def trimask(specular, diffuse):
    """
    Separates pixels into three categories:
        1) Directly visible from camera
        2) Mirror surface (including imperfect mirrors)
        3) Reflected objects
    """
    pass

def segment(cloud, masks):
    """
    Segments the point cloud into three categories:
        1) Directly visible from camera
        2) Mirror surface (including imperfect mirrors)
        3) Reflected objects

    Parameters
    ----------
    cloud : np.array[x, 3]
        The full point cloud.
    masks : dict of np.array[m, n]
        {
            'direct': Mask of pixels directly seen by the camera.
            'mirror': Mask of pixels corresponding to mirrors in the scene.
            'reflected': Mask of pixels known to be reflected through a mirror.
        }

    Returns
    -------
    direct : np.array[?, 3]
        Point cloud that requires no manipulation.
    mirror_planes : np.array[a, 3]
        Plane normals for mirrors.
    reflect : list of np.array[?, 3]; a entries in list
        Point clouds that will be reflected over the mirror.
        Split into as many distinct point clouds as there are mirror normals.
    """

    clouds = {key: list() for key in ['direct', 'mirror', 'reflect']}
    for key, mask in masks.items():

        clouds[key].append(point_segmentation(cloud, mask))

    direct = np.vstack(clouds['direct'])
    mirror = np.vstack(clouds['mirror'])
    reflect = np.vstack(clouds['reflect'])

    # TODO: split with clustering?
    mirror_planes = find_plane(mirror)

    return direct, mirror_planes, reflect