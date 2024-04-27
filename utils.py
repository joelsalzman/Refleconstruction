import cv2
import os
import torch

def imshow(*args, **kwargs):

    for i, img in enumerate(args):
        cv2.imshow(str(i), img, **kwargs)

def imwrite(*args, **kwargs):

    for i, img in enumerate(args):
        path = os.path.join('data', 'outputs', f'img_{i}.png')
        cv2.imwrite(path, img, **kwargs)

def to_ply(tensor, output_path):

    # Convert the PyTorch tensor to a numpy array
    points = tensor.detach().numpy()
    
    # Get the number of points
    num_points = points.shape[0]
    
    # Create the PLY content
    ply_content = [
        "ply",
        "format ascii 1.0",
        f"element vertex {num_points}",
        "property float x",
        "property float y",
        "property float z",
        "end_header"
    ]
    
    # Add the point data to the PLY content
    for point in points:
        ply_content.append(f"{point[0]} {point[1]} {point[2]}")
    
    # Join the PLY content into a single string
    ply_content = "\n".join(ply_content)
    
    # Write the PLY content to the output file
    with open(output_path, "w") as file:
        file.write(ply_content)