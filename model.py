import torch
import torch.nn as nn
# import mathutils
# from mathutils import Vector, Matrix
from sklearn.neighbors import BallTree
from load import create_dataloader
from scipy.spatial import ConvexHull, distance

class Reconstructor(nn.Module):

    def __init__(self):

        super().__init__()

        self.direct = None
        self.normal = nn.Parameter(torch.randn(6, dtype=torch.float32))
        
        self.dist_weight = 0.00001

    def set_direct(self, direct):
        
        self.direct = direct

    def set_reflect(self, cloud):

        assert self.direct is not None
        assert cloud.shape[1] == 3, cloud.shape

        direct_center = self.direct.mean(dim=0)
        reflect_center = cloud.mean(dim=0)
        centerpoint = (direct_center + reflect_center) / 2
        orientation = direct_center - reflect_center

        self.normal.data = torch.cat([centerpoint, orientation])
        assert self.normal.shape == (6,), self.normal.shape

    def reflect(self, points):

        # Extract the location and orientation components
        location = self.normal[:3]
        orientation = self.normal[3:]

        # Normalize the orientation vector
        orientation = (orientation / torch.norm(orientation)).view(1, 3)

        # Translate the points by subtracting the location
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
    
    def gaussian_curvature(self, points, k):

        # Number of points
        n = points.shape[0]
        m = self.direct.shape[0]
        
        # Create a list to store the Gaussian curvature values
        gaussian_curvatures = torch.zeros(n)
        mask = torch.zeros(n, dtype=torch.bool)
        
        # Compute the Gaussian curvature for each point
        for i in range(n):

            # Find the k nearest neighbors of the current point
            distances = torch.norm(points - points[i], dim=1)
            _, indices = torch.topk(distances, k+1, largest=False)

            # Only care about the points close to the other point cloud
            if torch.all(indices < m) | torch.all(indices >= m):
                continue
            mask[i] = True
            neighbors = points[indices[1:]]
            
            # Center the neighboring points
            centered_neighbors = neighbors - points[i]
            
            # Compute the covariance matrix of the centered neighbors
            cov_matrix = torch.matmul(centered_neighbors.T, centered_neighbors) / (k - 1)
            
            # Compute the eigenvalues and eigenvectors of the covariance matrix
            eigenvalues, _ = torch.linalg.eigh(cov_matrix)
            
            # Compute the Gaussian curvature
            gaussian_curvatures[i] = torch.prod(eigenvalues) / (torch.sum(eigenvalues)**2 + 1e-6)
        
        return torch.mean(gaussian_curvatures[mask], dtype=torch.float32)
    
    def neighbor_mse(self, cloud, k):

        points_np = cloud.detach().numpy()
        tree = BallTree(points_np)
    
        # Find the indices and distances of the top k nearest neighbors for each point
        distances, indices = tree.query(points_np, k=k+1)
        
        # Remove the first index and distance (which is the point itself)
        indices = indices[:, 1:]
        distances = distances[:, 1:]
        
        # Convert the indices and distances to PyTorch tensors
        indices = torch.from_numpy(indices)
        distances = torch.from_numpy(distances)
        
        # Gather the neighboring points
        neighbors = cloud[indices]
        
        # Repeat the points tensor to match the shape of neighbors
        repeated_points = cloud.unsqueeze(1).repeat(1, k, 1)
        
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

        dist_loss = self.neighbor_mse(cloud, k)
        curve_loss = self.gaussian_curvature(cloud, k)

        loss = (self.dist_weight * dist_loss) + ((1 - self.dist_weight) * curve_loss) * 1e4
        print(self.normal.grad)
        return loss
    
    @torch.no_grad()
    def test(self, cloud):
        return self(cloud)