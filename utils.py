import cv2
import os
import torch
import matplotlib.pyplot as plt

def imshow(img):

    if img.dtype == 'bool' and len(img.shape) == 3:
        img = img.any(axis=2).astype(int)

    plt.imshow(img)
    
def imshow_(image_array):
    
    fig = plt.figure(dpi=300)

    # Calculate the figure size in inches based on the image dimensions and DPI
    fig_width = 1280 / fig.dpi
    fig_height = 480 / fig.dpi
    fig.set_size_inches(fig_width, fig_height)
    
    # Display the image
    plt.imshow(image_array)
    
    # Remove the axis
    plt.axis("off")
    
    # Adjust the subplot parameters to minimize padding
    plt.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    # Show the plot
    plt.show()
    

def imwrite(*args, **kwargs):

    for i, img in enumerate(args):
        path = os.path.join('data', 'outputs', f'img_{i}.png')
        cv2.imwrite(path, img, **kwargs)

def to_ply(tensor, output_path):

    # Convert the PyTorch tensor to a numpy array
    if tensor.device != 'cpu':
        tensor = tensor.cpu()
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