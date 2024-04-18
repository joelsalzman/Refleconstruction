import cv2
import os

def imshow(*args, **kwargs):

    for i, img in enumerate(args):
        cv2.imshow(str(i), img, **kwargs)

def imwrite(*args, **kwargs):

    for i, img in enumerate(args):
        path = os.path.join('data', 'outputs', f'img_{i}.png')
        cv2.imwrite(path, img, **kwargs)
