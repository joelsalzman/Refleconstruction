import torch
import numpy as np
import cv2
import os
from transformers import SamModel, SamProcessor
from matplotlib import pyplot as plt
from load import load_rgb
from rgb_coords import get_point_from_image
from segment_rgb import segment_with_sam

def SAM_input(filepath, rs, profile, color_frame, depth_frame, test=False, output=True):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SamModel.from_pretrained("facebook/sam-vit-huge").to(device)
    processor = SamProcessor.from_pretrained("facebook/sam-vit-huge")

    colorizer = rs.colorizer()

    color_image = cv2.cvtColor(np.asanyarray(color_frame.get_data()), cv2.COLOR_BGR2RGB)
    
    colorized_depth = np.asanyarray(colorizer.colorize(depth_frame).get_data())

    images = np.hstack((color_image, colorized_depth))
    plt.imshow(images)
    # plt.show()

    # Segment oout the points or use known coords
    if test:
        point_obj = (448.07, 293.53)
        point_mirr = (80.54, 220.80)
        point_ref = (123.40, 250.67)

    else:
        print("### SELECT OBJECT ###")
        point_obj = get_point_from_image(color_image, "### SELECT OBJECT ###")
        print("### SELECT MIRROR ###")
        point_mirr = get_point_from_image(color_image, "### SELECT MIRROR ###")
        print("### SELECT REFLECTION ###")
        point_ref = get_point_from_image(color_image, "### SELECT REFLECTION ###")

    print("Segmenting objects...")
    obj_mask = segment_with_sam(
        model, processor, [[[point_obj[0], point_obj[1]]]], color_image
    )[0][0][0].numpy()
    print("Segmenting objects...")
    mirr_mask = segment_with_sam(
        model, processor, [[[point_mirr[0], point_mirr[1]]]], color_image
    )[0][0][
        2
    ].numpy()  # keep mask 2
    print("Segmenting objects...")
    ref_mask = segment_with_sam(
        model, processor, [[[point_ref[0], point_ref[1]]]], color_image
    )[0][0][0].numpy()
    print("Done segmenting...")

    ### After segmenting get the depthitre
    height, width = color_image.shape[:2]
    expected = 300
    aspect = width / height

    depth = np.asanyarray(depth_frame.get_data())
    depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()

    obj_depth = depth[obj_mask > 0].astype(float) * depth_scale
    mirror_depth = depth[mirr_mask > 0].astype(float) * depth_scale
    ref_depth = depth[ref_mask > 0].astype(float) * depth_scale
    # depth = depth * depth_scale
    mdist, _, _, _ = cv2.mean(obj_depth)
    odist, _, _, _ = cv2.mean(mirror_depth)
    rdist, _, _, _ = cv2.mean(ref_depth)

    print(f"Detected obj {mdist:.3} meters away.")
    print(f"Detected mirror {odist:.3} meters away.")
    print(f"Detected reflection {rdist:.3} meters away.")

    messages = [
        f"Detected obj {mdist:.3} meters away.",
        f"Detected mirror {odist:.3} meters away.",
        f"Detected reflection {rdist:.3} meters away.",
    ]

    # VISUALIZE MAPS
    fig, axes = plt.subplots(1, 4, figsize=(15, 5))

    axes[0].imshow(color_image)
    axes[0].set_title("Original Image")
    mask_list = [obj_mask, mirr_mask, ref_mask]

    for i, mask in enumerate(mask_list, start=1):
        overlayed_image = np.array(color_image).copy()

        overlayed_image[:, :, 0] = np.where(mask == 1, 255, overlayed_image[:, :, 0])
        # overlayed_image[:,:,1] = np.where(mask == 1, 0, overlayed_image[:,:,1])
        # overlayed_image[:,:,2] = np.where(mask == 1, 0, overlayed_image[:,:,2])

        axes[i].imshow(overlayed_image)
        axes[i].set_title(messages[i - 1])
    for ax in axes:
        ax.axis("off")
    plt.show()

    if output:
        basename = os.path.basename(filepath).split('.')[0]
        cv2.imwrite(
            os.path.join('data', 'segmented', f'{basename}_direct_mask.png'),
            (255 * obj_mask).astype('uint8')
        )
        cv2.imwrite(
            os.path.join('data', 'segmented', f'{basename}_mirror_mask.png'),
            (255 * mirr_mask).astype('uint8')
        )
        cv2.imwrite(
            os.path.join('data', 'segmented', f'{basename}_reflect_mask.png'),
            (255 * ref_mask).astype('uint8')
        )

    return obj_mask, mirr_mask, ref_mask
