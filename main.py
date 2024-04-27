import torch
from load import create_dataloader
from utils import to_ply
# from segment import find_plane
# from transform import reflect_points
from model import Reconstructor
from torch.utils.tensorboard import SummaryWriter

# if __name__ == '__main__':

#     point_cloud, faces = load_ply(r'data\parrot2.ply')

#     plane_normal = find_plane(point_cloud)
#     points = reflect_points(point_cloud, plane_normal)


if __name__ == '__main__':

    model = Reconstructor()
    optimizer = torch.optim.SGD(model.parameters(), lr=1)
    es = 3

    dataloader = create_dataloader(r'data\segmented')
    torch.autograd.set_detect_anomaly(True)
    # writer = SummaryWriter(log_dir='logs')

    # Training loop
    for batch in dataloader:

        direct = batch['direct_points']
        reflect = batch['reflect_points']
        # tv = batch['tv_points']
        cloud = None

        model.set_direct(direct)
        model.set_reflect(reflect)

        normals = list()
        for epoch in range(10):

            # Forward pass
            cloud = model(reflect)
            
            # Compute loss
            loss = model.loss(cloud)
            normal = model.normal.tolist()
            normals.append(normal)

            # Logging
            print(f'Epoch {epoch} Loss: {loss}')
            print(f'Location: {normal[:3]}')
            print(f'Rotation: {normal[3:]}\n')
            # writer.add_graph(model, reflect)
            if len(normals) > es and all([n == normal for n in normals[-es-1:-1]]):
                break
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # torch_to_blender(cloud)
        to_ply(model.reflect(reflect), r'data\outputs\parrot_flipped.ply')