# %% Setup
import torch
from transformers import SamModel, SamProcessor
from load import load_rgb
from matplotlib import pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"
model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

# %% Create masks
raw_image = load_rgb(r'data\parrot_test_5_Color.png')
seeds = [
    [560, 245],  # TV
    [295, 135],  # Parrot direct
    [513, 206],  # Parrot in TV
    [73,  215],  # Parrot in mirror
    [32,  290],  # Mirror (stand)
]

masks = list()
scores = list()
for pt in seeds:

    input_points = [[pt]]

    inputs = processor(raw_image, input_points=input_points, return_tensors="pt"
                       ).to(device)
    with torch.no_grad():
        outputs = model(**inputs)

    m = processor.image_processor.post_process_masks(
        outputs.pred_masks.cpu(),
        inputs["original_sizes"].cpu(),
        inputs["reshaped_input_sizes"].cpu()
    )
    s = outputs.iou_scores

    img = m[0].squeeze().permute(1, 2, 0).to(torch.float32)
    plt.imshow(img)

    masks.append(m)
    scores.append(s)
