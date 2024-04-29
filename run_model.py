import torch
from load import create_dataloader
from utils import to_ply
from model import Reconstructor
# from torch.utils.tensorboard import SummaryWriter
import numpy as np
import os

# if __name__ == '__main__':

#     point_cloud, faces = load_ply(r'data\parrot2.ply')

#     plane_normal = find_plane(point_cloud)
#     points = reflect_points(point_cloud, plane_normal)


def run_model(basename=None, sixdof=None):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = Reconstructor().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    es = 25

    dataloader = create_dataloader(r'data\segmented')
    torch.autograd.set_detect_anomaly(True)
    # writer = SummaryWriter(log_dir='logs')

    # Training loop
    for batch in dataloader:

        direct = batch['direct_points']
        reflected = batch['reflect_points']
        mirror = batch['mirror_points'] if 'mirror_points' in batch else None
        
        if basename is not None and batch['name'][0] != basename:
            print(f"Skipping {basename}")
            continue
        basename = batch['name']

        cloud = None

        model.set_direct(direct)
        
        if sixdof is not None:
            model.set_normal(sixdof)
        elif mirror is not None:
            model.initialize_by_mirror(mirror)
        else:
            model.initialize_by_centerpoint(reflected)

        to_ply(model.reflect(reflected), os.path.join(
            'data', 'outputs', f'{basename}_flipped_initial.ply')
        )

        sixdofs = list()
        losses = list()
        for epoch in range(10000):

            # Forward pass
            cloud = model(reflected)
            
            # Compute loss
            loss = model.loss(cloud)
            sixdof = model.sixdof.tolist()
            losses.append(loss)
            sixdofs.append(sixdof)

            # Logging
            print(f'Epoch {epoch} Loss: {loss}')
            print(f'Location: {sixdof[:3]}')
            print(f'Rotation: {sixdof[3:]}\n')
            # writer.add_graph(model, reflect)
            if len(sixdofs) > es:
                if all([loss >= prev_loss for prev_loss in losses[-es-1:-1]]):
                    nearby = losses[-es-1:-1]
                    if device != 'cpu':
                        nearby = [f.cpu() for f in nearby]
                    sixdof = sixdofs[np.argmin([f.detach().numpy() for f in nearby])]
                    model.set_normal(sixdof)
                    break
                elif all([n == sixdof for n in sixdofs[-es-1:-1]]):
                    break
            
            # Backprop
            optimizer.zero_grad()
            loss.backward()
            # if epoch % 2: # this is to prioritize rotation
            #     model.sixdof.grad[:3] = 0
            optimizer.step()

        # torch_to_blender(cloud)
        to_ply(model.reflect(reflected), os.path.join(
            'data', 'outputs', f'{basename}_flipped_final.ply')
        )

    return model

if __name__ == '__main__':
    run_model('cvbook')