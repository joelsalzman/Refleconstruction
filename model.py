import torch
import torch.nn as nn
# import mathutils
# from mathutils import Vector, Matrix
from sklearn.neighbors import BallTree
from scipy.spatial import ConvexHull, distance
import numpy as np

class Reconstructor(nn.Module):

    def __init__(self):

        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.direct = None
        self.apart = None
        self.sixdof = nn.Parameter(torch.randn(6, dtype=torch.float32).to(self.device))
        
        self.dist_weight = 0.1e-2

    def set_direct(self, direct):
        
        self.direct = direct.to(self.device)

    def set_normal(self, normal):

        assert isinstance(normal, list)
        self.sixdof.data = torch.tensor(normal).to(self.device)


    def initialize_by_centerpoint(self, cloud):
        """this one doesnt work well at all"""

        assert self.direct is not None
        assert cloud.shape[1] == 3, cloud.shape

        # We just find the vector between the centroids and use that
        direct_center = self.direct.mean(dim=0)
        reflect_center = cloud.to(self.device).mean(dim=0)
        startpoint = (direct_center + reflect_center) / 1.75
        orientation = direct_center - reflect_center

        self.sixdof.data = torch.cat([startpoint, orientation])
        assert self.sixdof.shape == (6,), self.sixdof.shape


    def initialize_by_mirror(self, cloud):

        # Center the points around the origin
        centroid = torch.mean(cloud, axis=0)
        centered_points = cloud - centroid

        # Need to have points in numpy
        if centered_points.device != 'cpu':
            centered_points = centered_points.cpu()
        points = centered_points.detach().numpy()

        # Use SVD to find a plane to the points and find the plane normal
        _, _, vh = np.linalg.svd(points)
        normal = torch.from_numpy(vh[-1]).to(self.device)
        self.sixdof.data = torch.cat([normal/torch.norm(normal), centroid])
        assert self.sixdof.data.shape == (6,)


    def reflect(self, points):

        # Extract the location and orientation components
        location = self.sixdof[:3]
        orientation = self.sixdof[3:]

        # Normalize the orientation vector
        orientation = (orientation / torch.norm(orientation)).view(1, 3)

        # Translate the points by subtracting the location
        if not points.device == self.device:
            points = points.to(self.device)
        translated_points = points - location

        # Compute the projection of translated points onto the orientation vector
        projection = torch.sum(translated_points * orientation, dim=1, keepdim=True) * orientation

        # Reflect the translated points by subtracting twice the projection
        reflected_points = translated_points - 2 * projection

        # Translate the reflected points back by adding the location
        final_points = reflected_points + location
        return final_points.to(torch.float32)

    
    def forward(self, points):

        assert self.direct is not None

        reflected = self.reflect(points)
        cloud = torch.cat([self.direct, reflected]).to(torch.float32)
        return cloud
    
    # def to_mathutils(self, torch_points):
    #     return [mathutils.Vector(pt.tolist()) for pt in torch_points]

    # def to_pytorch(self, bpy_points):
    #     return torch.tensor([[pt.x, pt.y, pt.z] for pt in bpy_points], dtype=torch.float32)

    def knn(self, cloud, k):

        points = cloud.cpu().detach().numpy()
        self.tree = BallTree(points)
        distances, indices = self.tree.query(points, k+1, dualtree=True)
        distances = torch.from_numpy(distances).to(self.device)
        indices = torch.from_numpy(indices).to(self.device)

        return distances, indices
    
    def gaussian_curvature(self, points, k, distances, indices):

        # Number of points
        n = points.shape[0]
        m = self.direct.shape[0]

        # Declare arrays for storing curvature values
        gaussian_curvatures = torch.zeros(n)
        
        # See if the point clouds are close
        mask = torch.tensor([
                ~(torch.all(indices[i] < m) | torch.all(indices[i] >= m))
                for i in range(n)
            ], dtype=torch.bool
        )

        # If the point clouds are too far apart, no need to waste time computing
        if not torch.any(mask):
            print('Clouds far apart')
            # self.apart = 1
            return
        print(f'Analyzing {mask.sum()}/{mask.shape[0]} nearby points')
        # self.apart = -1

        # Iterate through useful points
        for i in torch.argwhere(mask):

            # Find the k nearest neighbors of the current point
            neighbors = points[indices[i, :]].squeeze()
            
            # Center the neighboring points
            centered_neighbors = neighbors - points[i]
            
            # Compute the covariance matrix of the centered neighbors
            cov_matrix = torch.matmul(centered_neighbors.T, centered_neighbors) / (k - 1)
            
            # Compute the eigenvalues and eigenvectors of the covariance matrix
            eigenvalues, _ = torch.linalg.eigh(cov_matrix)
            
            # Compute the Gaussian curvature
            gaussian_curvatures[i] = torch.prod(eigenvalues) / (torch.sum(eigenvalues)**2 + 1e-6)
        
        squared_curvatures = torch.pow(gaussian_curvatures, 2)
        return torch.mean(squared_curvatures[mask], dtype=torch.float32) * 1e10
    
  
    def convex_hull(self, cloud):

        pass


    def chamfer_distance(self, p1, p2):
        """
        Compute the Chamfer distance between two point clouds.
        
        Args:
            p1 (torch.Tensor): First point cloud, shape (N, D)
            p2 (torch.Tensor): Second point cloud, shape (M, D)
        
        Returns:
            torch.Tensor: Chamfer distance between p1 and p2
        """
        # Compute pairwise distances between points in p1 and p2
        distances = torch.cdist(p1, p2)
        
        # Find the minimum distance for each point in p1 to p2
        min_dist_p1_to_p2, _ = torch.min(distances, dim=1)
        
        # Find the minimum distance for each point in p2 to p1
        min_dist_p2_to_p1, _ = torch.min(distances, dim=0)
        
        # Compute the Chamfer distance
        chamfer_dist = torch.mean(min_dist_p1_to_p2) + torch.mean(min_dist_p2_to_p1)
        
        return chamfer_dist
        
    def loss(self, cloud, k=10):

        distances, indices = self.knn(cloud, k)

        curve_loss = self.gaussian_curvature(cloud, k, distances, indices)
        dist_loss = self.chamfer_distance(self.direct, cloud[self.direct.shape[0]:])

        if curve_loss is not None:
            loss = (self.dist_weight * dist_loss) + ((1 - self.dist_weight) * curve_loss)
        else:
            loss = dist_loss

        return loss