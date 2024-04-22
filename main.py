from load import load_ply
from segment import find_plane
from transform import reflect_points

if __name__ == '__main__':

    point_cloud, faces = load_ply(r'data\parrot2.ply')

    plane_normal = find_plane(point_cloud)
    points = reflect_points(point_cloud, plane_normal)