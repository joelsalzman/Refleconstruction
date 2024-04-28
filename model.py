import torch
import torch.nn as nn
# import mathutils
# from mathutils import Vector, Matrix
from sklearn.neighbors import BallTree
from load import create_dataloader
from scipy.spatial import ConvexHull, distance
import numpy as np

class Reconstructor(nn.Module):

    def __init__(self):

        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.direct = None
        self.normal = nn.Parameter(torch.randn(6, dtype=torch.float32).to(self.device))
        
        self.dist_weight = 0.00001

    def set_direct(self, direct):
        
        self.direct = direct.to(self.device)

    def set_normal(self, normal):

        assert isinstance(normal, list)
        self.normal.data = torch(normal).to(self.device)

    def initialize_by_centerpoint(self, cloud):

        assert self.direct is not None
        assert cloud.shape[1] == 3, cloud.shape

        # We just find the vector between the centroids and use that
        direct_center = self.direct.mean(dim=0)
        reflect_center = cloud.to(self.device).mean(dim=0)
        startpoint = (direct_center + reflect_center) / 1.8
        orientation = direct_center - reflect_center

        self.normal.data = torch.cat([startpoint, orientation])
        assert self.normal.shape == (6,), self.normal.shape


    def initialize_by_mirror(self, cloud):

        # Center the points around the origin
        centroid = torch.mean(cloud, axis=0)
        centered_points = cloud - centroid

        # Need to be in numpy
        if centered_points.device != 'cpu':
            centered_points = centered_points.cpu()
        points = centered_points.detach().numpy()

        # Use SVD to find a plane to the points and find the plane normal
        _, _, vh = np.linalg.svd(points)
        normal = torch.from_numpy(vh[-1]).to(self.device)
        self.normal = normal/torch.norm(normal) + centroid


    def reflect(self, points):

        # Extract the location and orientation components
        location = self.normal[:3]
        orientation = self.normal[3:]

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
        mask = torch.zeros(n, dtype=torch.bool)
        
        # See if the point clouds are close
        for i in range(n):

            # Only care about the points close to the other point cloud
            if torch.all(indices[i] < m) | torch.all(indices[i] >= m):
                continue
            mask[i] = True

        # If the point clouds are too far apart, we should look at every point
        if not torch.any(mask):
            print('Clouds far apart')
            return torch.tensor([1]).to(self.device)
        else:
            print(f'Analyzing {mask.sum()}/{mask.shape[0]} nearby points')

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
        
        return torch.mean(gaussian_curvatures[mask], dtype=torch.float32)
    
    def neighbor_mse(self, cloud, k, distances, indices):

        # Gather the neighboring points
        neighbors = cloud[indices]
        
        # Repeat the points tensor to match the shape of neighbors
        repeated_points = cloud.unsqueeze(1).repeat(1, k+1, 1)
        
        # Compute the squared differences between points and their neighbors
        squared_diffs = torch.pow(repeated_points - neighbors, 2)
        
        # Compute the mean squared error loss
        mse_loss = torch.mean(squared_diffs, dtype=torch.float32)
        return mse_loss
    
    # def hull_distance(self, cloud):

    #     reflect = cloud[self.direct.shape[0]:]
        
    #     d_hull = ConvexHull(self.direct.detach().numpy())
    #     r_hull = ConvexHull(reflect.detach().numpy())

    #     return distance.cdist()
        
    def loss(self, cloud, k=10):

        distances, indices = self.knn(cloud, k)

        curve_loss = self.gaussian_curvature(cloud, k, distances, indices)
        dist_loss = self.neighbor_mse(cloud, k, distances, indices)

        loss = (self.dist_weight * dist_loss) + ((1 - self.dist_weight) * curve_loss)
        return loss
    
    @torch.no_grad()
    def test(self, cloud):
        return self(cloud)