#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
import torch
from PIL import Image
import numpy as np
import cv2

def mse(img1, img2):
    return (((img1 - img2)) ** 2).view(img1.shape[0], -1).mean(1, keepdim=True)

def save_img_torch(x, name='out.png'):
    x = (x.clamp(0., 1.).detach().cpu().numpy() * 255).astype(np.uint8)
    if x.shape[0] == 1 or x.shape[0] == 3:
        x = x.transpose(1, 2, 0)
    if x.shape[-1] == 1:
        x = x.squeeze(-1)

    img = Image.fromarray(x)
    img.save(name)

def visualize_depth_numpy(depth, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    depth: (H, W)
    """
    x = np.nan_to_num(depth) # change nan to 0
    if minmax is None:
        mi = np.max(np.min(x), 0) # get minimum positive depth (ignore background)
        ma = np.max(x)
    else:
        mi,ma = minmax
    x = (x-mi)/(ma-mi+1e-8) # normalize to 0~1
    x = (255*x).astype(np.uint8)
    x_ = cv2.applyColorMap(x, cmap)
    return x_, [mi,ma]

def visualize_normal_numpy(normal, minmax=None, cmap=cv2.COLORMAP_JET):
    """
    normal: (3, H, W)
    """
    if normal.shape[0] != 3:
        raise ValueError("Input normal must have 3 channels")


    H, W = normal.shape[1], normal.shape[2]
    result = np.zeros((H, W, 3), dtype=np.uint8)

    for i in range(3):
        x = np.nan_to_num(normal[i])  # change nan to 0
        if minmax is None:
            mi = np.min(x)  # get minimum positive depth (ignore background)
            ma = np.max(x)
        else:
            mi, ma = minmax
        x = (x - mi) / (ma - mi + 1e-8)  # normalize to 0~1
        x = (255 * x).astype(np.uint8)
        x_colored = cv2.applyColorMap(x, cmap)
        result[:, :, i] = x_colored[:, :, 0]

    return result, [mi, ma]
