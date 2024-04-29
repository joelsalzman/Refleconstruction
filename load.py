import numpy as np
import open3d as o3d
import cv2
import torch
import os
# import bpy

def load_ply(file_path):

    mesh = o3d.io.read_triangle_mesh(file_path)
    vertices = np.asarray(mesh.vertices)
    return vertices

def load_rgb(file_path):

    img = cv2.imread(file_path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    

class PLYDataset(torch.utils.data.Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.file_groups = self._get_file_groups()

    def _get_file_groups(self):
        file_groups = {}
        for file in os.listdir(self.root_dir):
            if file.endswith('.ply'):
                key = file.rsplit('_', 1)[0]
                if key not in file_groups:
                    file_groups[key] = {}
                if '_direct.ply' in file:
                    file_groups[key]['direct'] = os.path.join(self.root_dir, file)
                elif '_reflect.ply' in file:
                    file_groups[key]['reflect'] = os.path.join(self.root_dir, file)
                elif '_mirror.ply' in file:
                    file_groups[key]['mirror'] = os.path.join(self.root_dir, file)
        return file_groups

    def __len__(self):
        return len(self.file_groups)

    def __getitem__(self, index):
        key = list(self.file_groups.keys())[index]
        file_group = self.file_groups[key]

        direct_points = None
        reflect_points = None
        mirror_points = None

        if 'direct' in file_group:
            pcd = o3d.io.read_point_cloud(file_group['direct'])
            direct_points = np.asarray(pcd.points)

        if 'reflect' in file_group:
            pcd = o3d.io.read_point_cloud(file_group['reflect'])
            reflect_points = np.asarray(pcd.points)

        if 'mirror' in file_group:
            pcd = o3d.io.read_point_cloud(file_group['mirror'])
            mirror_points = np.asarray(pcd.points)

        return direct_points, reflect_points, mirror_points

def collate_fn(batch):
    direct_points_list = []
    reflect_points_list = []
    mirror_points_list = []

    for direct_points, reflect_points, mirror_points in batch:
        if direct_points is not None:
            direct_points_list.append(direct_points)
        if reflect_points is not None:
            reflect_points_list.append(reflect_points)
        if mirror_points is not None:
            mirror_points_list.append(mirror_points)

    collated_batch = {}

    if direct_points_list:
        num_points_list = [len(points) for points in direct_points_list]
        max_points = max(num_points_list)
        padded_points = []
        for points in direct_points_list:
            pad_size = max_points - len(points)
            padded = np.pad(points, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
            padded_points.append(padded)
        collated_batch['direct_points'] = torch.from_numpy(np.stack(padded_points, axis=0)).float().squeeze()
        collated_batch['direct_num_points'] = torch.tensor(num_points_list, dtype=torch.long).squeeze()

    if reflect_points_list:
        num_points_list = [len(points) for points in reflect_points_list]
        max_points = max(num_points_list)
        padded_points = []
        for points in reflect_points_list:
            pad_size = max_points - len(points)
            padded = np.pad(points, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
            padded_points.append(padded)
        collated_batch['reflect_points'] = torch.from_numpy(np.stack(padded_points, axis=0)).float().squeeze()
        collated_batch['reflect_num_points'] = torch.tensor(num_points_list, dtype=torch.long).squeeze()

    if mirror_points_list:
        num_points_list = [len(points) for points in mirror_points_list]
        max_points = max(num_points_list)
        padded_points = []
        for points in mirror_points_list:
            pad_size = max_points - len(points)
            padded = np.pad(points, ((0, pad_size), (0, 0)), mode='constant', constant_values=0)
            padded_points.append(padded)
        collated_batch['mirror_points'] = torch.from_numpy(np.stack(padded_points, axis=0)).float().squeeze()
        collated_batch['mirror_num_points'] = torch.tensor(num_points_list, dtype=torch.long).squeeze()

    return collated_batch

def create_dataloader(root_dir, batch_size=1, shuffle=True, num_workers=0):
    
    dataset = PLYDataset(root_dir)
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle,
        num_workers=num_workers, 
        collate_fn=collate_fn
    )
    return dataloader

