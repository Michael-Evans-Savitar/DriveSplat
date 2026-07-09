#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
import torch
from einops import repeat
from utils.sh_utils import eval_sh
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model_all import DriveSplatModel
from utils.general_utils import quaternion_to_matrix
import os
from concurrent.futures import ThreadPoolExecutor

class PrefilterVoxel:
    def __init__(self, pipe=None, bg_color=None, scaling_modifier = 1.0, override_color = None):
        self.pipe = pipe
        self.bg_color = bg_color
        self.scaling_modifier = scaling_modifier
        self.override_color = override_color

    def prefilter_voxel(self, viewpoint_camera, pc : DriveSplatModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):
        results = []
        for model_name in pc.graph_gaussian_range.keys():
            single_model = getattr(pc, model_name)
            if model_name == "background":
                result = self.prefilter_voxel_kernel(viewpoint_camera, single_model, pipe, bg_color, scaling_modifier, override_color, "bkgd")
            else:
                result = self.prefilter_voxel_kernel(viewpoint_camera, single_model, pipe, bg_color, scaling_modifier, override_color, "obj")
            single_model.visible_mask = result
            results.append(result)
        results = torch.cat(results, dim=0)

        return results

    def prefilter_obj_voxel(self, viewpoint_camera, pc : DriveSplatModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None):

        results = self.obj_prefilter_voxel_kernel(viewpoint_camera, pc, pipe, bg_color, scaling_modifier, override_color)

        return results

    def prefilter_voxel_kernel(self, viewpoint_camera, pc, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
                               override_color=None, obj="obj"):
        """
        Render the scene for a single DriveSplatModel instance.
        """
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=1,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)


        if pc.model_name == "background":

            means3D = pc.get_anchor[pc._anchor_mask]


            if means3D.shape[0] == 0:
                print(f"[WARNING] Background model {pc.model_name}: No points in _anchor_mask, returning empty mask")
                visible_mask = pc._anchor_mask.clone()
                return visible_mask

            scales = None
            rotations = None
            cov3D_precomp = None
            if pipe.compute_cov3D_python:
                cov3D_precomp = pc.get_covariance(scaling_modifier)
            else:
                scales = pc.get_scaling[pc._anchor_mask]
                rotations = pc.get_rotation[pc._anchor_mask]

            radii_pure = rasterizer.visible_filter(means3D = means3D,
                scales = scales[:,:3] if scales is not None else None,
                rotations = rotations,
                cov3D_precomp = cov3D_precomp)

            visible_mask = pc._anchor_mask.clone()
            if isinstance(radii_pure, tuple):
                visible_mask[pc._anchor_mask] = radii_pure[0] > 0
            else:
                visible_mask[pc._anchor_mask] = radii_pure > 0

            return visible_mask

        elif pc.obj_class == 'pedestrian':

            visible_mask = pc._anchor_mask.clone()
            visible_mask[pc._anchor_mask] = True
            visible_mask[~pc._anchor_mask] = True
            return visible_mask

        else:

            try:

                obj_rot = quaternion_to_matrix(pc.obj_rots)
                temp = torch.einsum('bij, bj -> bi', obj_rot, pc.get_anchor) + pc.obj_trans
                means3D = temp[pc._anchor_mask]


                if means3D.shape[0] == 0:
                    visible_mask = pc._anchor_mask.clone()
                    visible_mask[:] = True
                    return visible_mask


                scales = None
                rotations = None
                cov3D_precomp = None
                if pipe.compute_cov3D_python:
                    cov3D_precomp = pc.get_covariance(scaling_modifier)
                else:
                    scales = pc.get_scaling[pc._anchor_mask]
                    rotations = pc.get_rotation[pc._anchor_mask]


                radii_pure = rasterizer.visible_filter(
                    means3D=means3D,
                    scales=scales[:, :3] if scales is not None else None,
                    rotations=rotations if rotations is not None else None,
                    cov3D_precomp=cov3D_precomp if cov3D_precomp is not None else None
                )
                visible_mask = pc._anchor_mask.clone()
                if isinstance(radii_pure, tuple):
                    visible_mask[pc._anchor_mask] = radii_pure[0] > 0
                else:
                    visible_mask[pc._anchor_mask] = radii_pure > 0
                return visible_mask

            except Exception as e:
                print(f"Error processing {pc.model_name}: {str(e)}")
                visible_mask = pc._anchor_mask.clone()
                visible_mask[:] = True
                return visible_mask

    def obj_prefilter_voxel_kernel(self, viewpoint_camera, pc, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
                                       override_color=None):
        """
        Render the scene for a single DriveSplatModel instance.
        """

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)
        bg_depth = torch.tensor([0]).float().cuda()

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=1,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        means3D = pc.get_anchor[pc.obj_anchor_mask]

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling[pc.obj_anchor_mask]
            rotations = pc.get_rotation[pc.obj_anchor_mask]

        # Perform visibility filtering
        radii_pure = rasterizer.visible_filter(
            means3D=means3D,
            scales=scales[:, :3] if scales is not None else None,
            rotations=rotations if rotations is not None else None,
            cov3D_precomp=cov3D_precomp if cov3D_precomp is not None else None
        )

        visible_mask = pc.obj_anchor_mask.clone()
        #  radii_pure
        if isinstance(radii_pure, tuple):
            visible_mask[pc.obj_anchor_mask] = radii_pure[0] > 0
        else:
            visible_mask[pc.obj_anchor_mask] = radii_pure > 0
        if visible_mask.sum() == 0:
            print(f"for voxel prefilter, model {pc.model_name} visible_mask is {visible_mask}")
        return visible_mask

    def obj_prefilter_voxel_kernel_single(self, viewpoint_camera, pc, pipe, bg_color: torch.Tensor, scaling_modifier=1.0,
                                       override_color=None):
        """
        Render the scene for a single DriveSplatModel instance.
        """
        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=1,
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )
        rasterizer = GaussianRasterizer(raster_settings=raster_settings)
        means3D = pc.get_anchor[pc._anchor_mask]

        # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
        # scaling / rotation by the rasterizer.
        scales = None
        rotations = None
        cov3D_precomp = None
        if pipe.compute_cov3D_python:
            cov3D_precomp = pc.get_covariance(scaling_modifier)
        else:
            scales = pc.get_scaling[pc._anchor_mask]
            rotations = pc.get_rotation[pc._anchor_mask]

        # Perform visibility filtering
        radii_pure = rasterizer.visible_filter(
            means3D=means3D,
            scales=scales[:, :3] if scales is not None else None,
            rotations=rotations if rotations is not None else None,
            cov3D_precomp=cov3D_precomp if cov3D_precomp is not None else None
        )

        visible_mask = pc._anchor_mask.clone()
        #  radii_pure
        if isinstance(radii_pure, tuple):
            visible_mask[pc._anchor_mask] = radii_pure[0] > 0
        else:
            visible_mask[pc._anchor_mask] = radii_pure > 0



        #     # _anchor_maskfallback

        return visible_mask
