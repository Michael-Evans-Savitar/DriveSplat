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

def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R

def generate_neural_gaussians(pipe, viewpoint_camera, pc : DriveSplatModel, visible_mask=None, is_training=False,  ape_code=-1, stage="fine"):
    ## view frustum filtering for acceleration
    if visible_mask is None:
        visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device = pc.get_anchor.device)

    anchor = pc.get_anchor[visible_mask]
    feat = pc.get_anchor_feat[visible_mask]
    level = pc.get_level[visible_mask]
    grid_offsets = pc._offset[visible_mask]
    grid_scaling = pc.get_scaling[visible_mask]

    ## get view properties for anchor
    ob_view = anchor - viewpoint_camera.camera_center
    # dist
    ob_dist = ob_view.norm(dim=1, keepdim=True)
    # view
    ob_view = ob_view / ob_dist

    ## view-adaptive feature
    if pc.use_feat_bank:
        if pc.add_level:
            cat_view = torch.cat([ob_view, level], dim=1)
        else:
            cat_view = ob_view

        bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1)

        ## multi-resolution feat
        feat = feat.unsqueeze(dim=-1)
        feat = feat[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
            feat[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
            feat[:,::1, :1]*bank_weight[:,:,2:]
        feat = feat.squeeze(dim=-1)

    if pc.add_level:
        cat_local_view = torch.cat([feat, ob_view, ob_dist, level], dim=1)
        cat_local_view_wodist = torch.cat([feat, ob_view, level], dim=1)
    else:
        cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)
        cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)

    if pc.appearance_dim > 0:
        if is_training or ape_code < 0:
            camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * viewpoint_camera.uid
            appearance = pc.get_appearance(camera_indicies)
        else:
            camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=ob_dist.device) * ape_code[0]
            appearance = pc.get_appearance(camera_indicies)

    # get offset's opacity
    if pc.add_opacity_dist:
        neural_opacity = pc.get_opacity_mlp(cat_local_view)
    else:
        neural_opacity = pc.get_opacity_mlp(cat_local_view_wodist)

    if pc.dist2level=="progressive":
        prog = pc._prog_ratio[visible_mask]
        transition_mask = pc.transition_mask[visible_mask]
        prog[~transition_mask] = 1.0
        neural_opacity = neural_opacity * prog

    # opacity mask generation
    neural_opacity = neural_opacity.reshape([-1, 1])
    mask = (neural_opacity>0.0)
    mask = mask.view(-1)

    # select opacity
    opacity = neural_opacity[mask]

    # get offset's color
    if pc.appearance_dim > 0:
        if pc.add_color_dist:
            color = pc.get_color_mlp(torch.cat([cat_local_view, appearance], dim=1))
        else:
            color = pc.get_color_mlp(torch.cat([cat_local_view_wodist, appearance], dim=1))
    else:
        if pc.add_color_dist:
            color = pc.get_color_mlp(cat_local_view)
        else:
            color = pc.get_color_mlp(cat_local_view_wodist)
    color = color.reshape([anchor.shape[0]*pc.n_offsets, 3])# [mask]

    # get offset's cov
    if pc.add_cov_dist:
        scale_rot = pc.get_cov_mlp(cat_local_view)
    else:
        scale_rot = pc.get_cov_mlp(cat_local_view_wodist)
    scale_rot = scale_rot.reshape([anchor.shape[0]*pc.n_offsets, 7]) # [mask]

    # offsets
    offsets = grid_offsets.view([-1, 3]) # [mask]

    # combine for parallel masking
    concatenated = torch.cat([grid_scaling, anchor], dim=-1)
    concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
    masked = concatenated_all[mask]
    scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)

    # post-process cov
    scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3]) # * (1+torch.sigmoid(repeat_dist))
    rot = pc.rotation_activation(scale_rot[:,3:7])

    # post-process offsets to get centers for gaussians
    offsets = offsets * scaling_repeat[:,:3]
    xyz = repeat_anchor + offsets

    rotations_mat = quaternion_to_matrix(rot)
    min_scales = torch.argmin(scaling, dim=-1)
    device = min_scales.device
    indices = torch.arange(min_scales.shape[0], device=device)
    normals = rotations_mat[indices, :, min_scales]
    dir_pp = (xyz - viewpoint_camera.camera_center.repeat(xyz.shape[0], 1))
    dir_pp_norm = dir_pp.norm(dim=1, keepdim=True)
    dir_pp_normalized = dir_pp / (dir_pp_norm + 1e-8)
    dotprod = torch.sum(-dir_pp_normalized * normals, dim=1, keepdim=True)
    normals = torch.where(dotprod >= 0, normals, -normals)

    semantic = pc._semantic[visible_mask]
    concatenated_semantic = repeat(semantic, 'n (c) -> (n k) (c)', k=pc.n_offsets)
    semantics = concatenated_semantic[mask]
    # (gaussian16 = (max_sh_degree+1)^2)
    shs_features = None  # , N * 16 * 3 * 4 bytes

    if is_training:
        return xyz, color, opacity, scaling, rot, neural_opacity, mask, normals, semantics, shs_features
    else:
        return xyz, color, opacity, scaling, rot, normals, semantics, shs_features

def render(viewpoint_camera, pc : DriveSplatModel, pipe, hyper, bg_color : torch.Tensor, scaling_modifier=1.0, visible_mask=None, stage = "fine", retain_grad=False, ape_code=-1, override_color=None):
    include_list = list(set(pc.model_name_id.keys()))

    # Step1: render foreground
    pc.set_visibility(include_list)
    result = render_kernel(viewpoint_camera, pc, pipe, hyper, bg_color, scaling_modifier, visible_mask, stage, retain_grad, ape_code, override_color)

    return result

def render_kernel(viewpoint_camera, pc : DriveSplatModel, pipe, hyper, bg_color : torch.Tensor, scaling_modifier=1.0, visible_mask=None, stage = "fine", retain_grad=False, ape_code=-1, override_color=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """
    include_list = list(set(pc.model_name_id.keys()))
    pc.set_visibility(include_list)
    is_training = pc.get_color_mlp.training

    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask, normals, semantics, shs_features = generate_neural_gaussians(pipe, viewpoint_camera, pc, visible_mask, is_training=is_training, stage=stage)
    else:
        xyz, color, opacity, scaling, rot, normals, semantics, shs_features = generate_neural_gaussians(pipe, viewpoint_camera, pc, visible_mask, is_training=is_training, ape_code=ape_code, stage=stage)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

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
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = scaling
        rotations = rot

    shs = shs_features
    time = torch.tensor(viewpoint_camera.time).to(xyz.device).repeat(xyz.shape[0], 1)
    if "coarse" in stage:
        xyz, scaling, rot, opacity, shs = xyz, scales, rotations, opacity, shs
    elif "fine" in stage:
        xyz, scaling, rot, opacity, shs = xyz, scales, rotations, opacity, shs
    else:
        raise NotImplementedError

    feature_names = []
    feature_dims = []
    features = []

    if pipe.render_normal:
        feature_names.append('normals')
        feature_dims.append(normals.shape[-1])
        features.append(normals)

    if hyper.use_semantic:
        feature_names.append('semantic')
        feature_dims.append(semantics.shape[-1])
        features.append(semantics)

    if len(features) > 0:
        features = torch.cat(features, dim=-1)
    else:
        features = None

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, depth, rendered_feature = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color, #color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None,
        semantics = features)

    rendered_feature_dict = dict()
    if rendered_feature.shape[0] > 0:
        rendered_feature_list = torch.split(rendered_feature, feature_dims, dim=0)
        for i, feature_name in enumerate(feature_names):
            rendered_feature_dict[feature_name] = rendered_feature_list[i]
    if 'normals' in rendered_feature_dict:
        rendered_feature_dict['normals'] = torch.nn.functional.normalize(rendered_feature_dict['normals'], dim=0)
    if 'semantic' in rendered_feature_dict:
        rendered_semantic = rendered_feature_dict['semantic']
        semantic_mode = hyper.semantic_mode
        if semantic_mode == 'logits':
            pass
        else:
            rendered_semantic = rendered_semantic / (torch.sum(rendered_semantic, dim=0, keepdim=True) + 1e-8) # normalize to probabilities
            rendered_semantic = torch.log(rendered_semantic + 1e-8) # change for cross entropy loss

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        result = {"render": rendered_image,
                "rendered_depth": depth,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                }
    else:
        result = {"render": rendered_image,
                "rendered_depth": depth,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                }
    result.update(rendered_feature_dict)
    return result

def render_object(viewpoint_camera, pc : DriveSplatModel, pipe, hyper, bg_color : torch.Tensor, scaling_modifier=1.0, visible_mask=None, stage = "fine", retain_grad=False, ape_code=-1, override_color=None):
    """
    Render the scene.

    Background tensor (bg_color) must be on GPU!
    """

    is_training = pc.get_color_mlp.training

    if is_training:
        xyz, color, opacity, scaling, rot, neural_opacity, mask, normals, semantics, shs_features = generate_neural_gaussians(pipe, viewpoint_camera, pc, visible_mask, is_training=is_training, stage=stage)
    else:
        xyz, color, opacity, scaling, rot, normals, semantics, shs_features = generate_neural_gaussians(pipe, viewpoint_camera, pc, visible_mask, is_training=is_training, ape_code=ape_code, stage=stage)

    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(xyz, dtype=pc.get_anchor.dtype, requires_grad=True, device="cuda") + 0
    if retain_grad:
        try:
            screenspace_points.retain_grad()
        except:
            pass

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
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)

    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = scaling
        rotations = rot

    shs = shs_features
    time = torch.tensor(viewpoint_camera.time).to(xyz.device).repeat(xyz.shape[0], 1)
    if "coarse" in stage:
        xyz, scaling, rot, opacity, shs = xyz, scales, rotations, opacity, shs
    elif "fine" in stage:
        xyz, scaling, rot, opacity, shs = xyz, scales, rotations, opacity, shs
    else:
        raise NotImplementedError

    feature_names = []
    feature_dims = []
    features = []

    if pipe.render_normal:
        feature_names.append('normals')
        feature_dims.append(normals.shape[-1])
        features.append(normals)

    if hyper.use_semantic:
        feature_names.append('semantic')
        feature_dims.append(semantics.shape[-1])
        features.append(semantics)

    if len(features) > 0:
        features = torch.cat(features, dim=-1)
    else:
        features = None

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, radii, depth, rendered_feature = rasterizer(
        means3D = xyz,
        means2D = screenspace_points,
        shs = None,
        colors_precomp = color, #color,
        opacities = opacity,
        scales = scaling,
        rotations = rot,
        cov3D_precomp = None,
        semantics = features)

    rendered_feature_dict = dict()
    if rendered_feature.shape[0] > 0:
        rendered_feature_list = torch.split(rendered_feature, feature_dims, dim=0)
        for i, feature_name in enumerate(feature_names):
            rendered_feature_dict[feature_name] = rendered_feature_list[i]
    if 'normals' in rendered_feature_dict:
        rendered_feature_dict['normals'] = torch.nn.functional.normalize(rendered_feature_dict['normals'], dim=0)
    if 'semantic' in rendered_feature_dict:
        rendered_semantic = rendered_feature_dict['semantic']
        semantic_mode = hyper.semantic_mode
        if semantic_mode == 'logits':
            pass
        else:
            rendered_semantic = rendered_semantic / (torch.sum(rendered_semantic, dim=0, keepdim=True) + 1e-8) # normalize to probabilities
            rendered_semantic = torch.log(rendered_semantic + 1e-8) # change for cross entropy loss

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    if is_training:
        result = {"render": rendered_image,
                "rendered_depth": depth,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                "selection_mask": mask,
                "neural_opacity": neural_opacity,
                "scaling": scaling,
                }
    else:
        result = {"render": rendered_image,
                "rendered_depth": depth,
                "viewspace_points": screenspace_points,
                "visibility_filter" : radii > 0,
                "radii": radii,
                }
    result.update(rendered_feature_dict)
    return result
