#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixK
import torch.nn.functional as F
from utils.general_utils import NumpytoTorch
from PIL import Image

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, K, image, gt_alpha_mask,
                 image_name, resolution_scale, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", depth = None, normal_gt = None, masks = dict(), time = 0, metadata = dict()
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.K = K
        self.image_name = image_name
        self.resolution_scale = resolution_scale
        self.time = time
        self.meta = metadata

        for name, mask in masks.items():
            setattr(self, name, mask)

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if 'ego_pose' in self.meta.keys():
            self.ego_pose = torch.from_numpy(self.meta['ego_pose']).float().cuda()
            del self.meta['ego_pose']
        if 'extrinsic' in self.meta.keys():
            self.extrinsic = torch.from_numpy(self.meta['extrinsic']).float().cuda()
            del self.meta['extrinsic']

        if 'axis_transform' in self.meta.keys():
            self.axis_transform = torch.from_numpy(self.meta['axis_transform']).float().cuda()
            del self.meta['axis_transform']

        if depth is not None:
            self.depth = depth.to(self.data_device)
        else:
            image_tensor = self.image.clone()  # self.image
            self.depth = self.estimate_depth_with_depth_anything(image_tensor)

        if normal_gt is not None:
            self.normal_gt = normal_gt.to(self.data_device)
        else:
            self.normal_gt = torch.zeros((self.image_height, self.image_width, 3), dtype=torch.float32, device=data_device)

        if 'lidar_depth' in self.meta.keys():
            # numpytorch tensorresize
            resolution = (self.image_width, self.image_height)
            lidar_depth = NumpytoTorch(self.meta['lidar_depth'], resolution, resize_mode=Image.NEAREST)
            # # batchresize
            self.lidar_depth = lidar_depth.to(self.data_device)
            del self.meta['lidar_depth']

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        if self.K is not None and K.size != 4:
            self.projection_matrix = getProjectionMatrixK(znear=self.znear, zfar=self.zfar, K=self.K, H=self.image_height, W=self.image_width).transpose(0,1).cuda()
            self.K = torch.from_numpy(self.K).float().cuda()
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

    def estimate_depth_with_depth_anything(self, image_tensor):
        #  (B, C, H, W)
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)

        # Depth Anything()
        if not hasattr(self, 'depth_model'):
            from depth_anything.dpt import DepthAnything
            self.depth_model = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14')
            self.depth_model.to(self.data_device)
            self.depth_model.eval()


        with torch.no_grad():
            depth = self.depth_model(image_tensor)


        depth = F.interpolate(
            depth,
            size=(self.image_height, self.image_width),
            mode='bilinear',
            align_corners=False
        )

        return depth

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform, time):
        self.image_width = width
        self.image_height = height
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]
        self.time = time
