#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
from scene.cameras import Camera
import numpy as np
from utils.general_utils import PILtoTorch, NumpytoTorch
from utils.graphics_utils import fov2focal
import torch
from PIL import Image
import torch.nn as nn
import copy
import torch.nn.functional as F

WARNED = False

def loadCam(args, id, cam_info, resolution_scale):
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        scale = float(resolution_scale * args.resolution)
        resolution = round(orig_w / scale), round(orig_h / scale)
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                global WARNED
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    new_w, new_h = resolution
    depth = cam_info.depth
    depth = NumpytoTorch(depth, resolution, resize_mode=Image.NEAREST)

    if cam_info.normal_gt is not None:
        normal_gt = PILtoTorch(cam_info.normal_gt, resolution)
    mask = loadmask(cam_info, resolution, resize_mode=Image.NEAREST)
    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]
    metadata = cam_info.metadata

    K = copy.deepcopy(cam_info.K)
    K[:2] /= scale

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T,
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, K=K,
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, resolution_scale=resolution_scale,
                  uid=id, data_device=args.data_device, depth=depth, normal_gt=normal_gt, masks=mask, time = cam_info.time, metadata=metadata)

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list

def camera_to_JSON(id, camera : Camera):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry

def loadmask(cam_info, resolution, resize_mode):
    masks = dict()
    if cam_info.semantic_gt is not None:
        masks['original_mask'] = PILtoTorch(cam_info.semantic_gt, resolution).clamp(0, 1).bool()
    #     masks['original_mask'] = None

    if 'sky_mask' in cam_info.metadata:
        masks['original_sky_mask'] = PILtoTorch(cam_info.metadata['sky_mask'], resolution).clamp(0, 1).bool()
        del cam_info.metadata['sky_mask']

    if 'obj_bound' in cam_info.metadata:
        masks['original_obj_bound'] = PILtoTorch(cam_info.metadata['obj_bound'], resolution).clamp(0, 1).bool()
        del cam_info.metadata['obj_bound']

    return masks
