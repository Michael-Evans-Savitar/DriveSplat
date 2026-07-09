#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
import torch
from einops import repeat
from scene.gaussian_model_all import DriveSplatModel
from utils.general_utils import quaternion_to_matrix

class GenerateNeuralGaussians:
    def __init__(self, pipe):
        self.pipe = pipe

    @staticmethod
    def apply_lod_opacity_adjustment(neural_opacity, pc, visible_mask):
        """
        Apply the learned LOD opacity bias from AGD.

        Args:
            neural_opacity: opacity tensor, either per-anchor or per-Gaussian.
            pc: Gaussian model instance.
            visible_mask: anchor visibility mask.

        Returns:
            Adjusted neural opacity tensor.
        """
        if not hasattr(pc, 'use_lod_bias') or not pc.use_lod_bias:
            return neural_opacity
        if not hasattr(pc, '_lod_bias') or pc._lod_bias is None:
            return neural_opacity

        n_anchors = pc._anchor.shape[0]
        n_bias = pc._lod_bias.shape[0]

        if n_bias != n_anchors:
            # Keep restored checkpoints compatible if the bias table size differs.
            if n_bias < n_anchors:
                padding = torch.zeros(n_anchors - n_bias,
                                     device=pc._lod_bias.device,
                                     dtype=pc._lod_bias.dtype)
                lod_bias_full = torch.cat([pc._lod_bias.detach(), padding], dim=0)
            else:
                lod_bias_full = pc._lod_bias[:n_anchors].detach()
        else:
            lod_bias_full = pc._lod_bias

        lod_bias_visible = lod_bias_full[visible_mask]  # [N_visible_anchors]
        opacity_scale_per_anchor = torch.sigmoid(lod_bias_visible)
        n_visible = opacity_scale_per_anchor.shape[0]

        neural_opacity_first_dim = neural_opacity.shape[0]
        expected_per_gaussian_size = n_visible * pc.n_offsets
        expected_per_anchor_size = n_visible


        if neural_opacity_first_dim == expected_per_anchor_size:
            if neural_opacity.dim() == 2 and opacity_scale_per_anchor.dim() == 1:
                opacity_scale_per_anchor = opacity_scale_per_anchor.unsqueeze(1)
            return neural_opacity * opacity_scale_per_anchor

        elif neural_opacity_first_dim == expected_per_gaussian_size:
            opacity_scale_expanded = opacity_scale_per_anchor.unsqueeze(1).expand(-1, pc.n_offsets).reshape(-1)
            if neural_opacity.dim() == 2:
                opacity_scale_expanded = opacity_scale_expanded.unsqueeze(1)
            return neural_opacity * opacity_scale_expanded

        else:

            print(f"[Warning] Opacity scale size mismatch: neural_opacity.shape={neural_opacity.shape}, "
                  f"expected_per_gaussian={expected_per_gaussian_size}, expected_per_anchor={expected_per_anchor_size}")
            return neural_opacity

    def generate_neural_gaussians(self, pipe, viewpoint_camera, pc : DriveSplatModel, visible_mask=None, is_training=False,  ape_code=-1, stage="fine"):
        offset_collect = []
        semantic_collect = []
        if pc.get_visibility('background'):
            bkgd_offset = pc.background._offset
            offset_collect.append(bkgd_offset)
            bkgd_semantic = pc.background._semantic
            semantic_collect.append(bkgd_semantic)
        for i, obj_name in enumerate(pc.graph_obj_list):
            obj_model = getattr(pc, obj_name)
            obj_offset = obj_model._offset
            offset_collect.append(obj_offset)
            obj_semantic = obj_model._semantic
            semantic_collect.append(obj_semantic)

        if len(offset_collect) > 0:
            offsets = torch.cat(offset_collect, dim=0)
            pc._offset = offsets
        if len(semantic_collect) > 0:
            semantics = torch.cat(semantic_collect, dim=0)
            pc._semantic = semantics

        if is_training:
            xyzs, colors, opacities, scalings, rots, neural_opacities, masks, normals, semantics, shs_features = self.generate_neural_gaussians_kernel(pipe, viewpoint_camera, pc, visible_mask, is_training, ape_code, stage)
        else:
            xyzs, colors, opacities, scalings, rots, normals, semantics, shs_features = self.generate_neural_gaussians_kernel(pipe, viewpoint_camera, pc, visible_mask, is_training, ape_code, stage)

        if is_training:
            return xyzs, colors, opacities, scalings, rots, neural_opacities, masks, normals, semantics, shs_features
        else:
            return xyzs, colors, opacities, scalings, rots, normals, semantics, shs_features

    def generate_neural_gaussians_kernel(self, pipe, viewpoint_camera, pc, visible_mask=None, is_training=False,  ape_code=-1, stage="fine"):
        """
        Generate anchor Gaussian attributes for the background/static model.
        """
        device = pc.get_anchor.device


        if visible_mask is None:
            visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device=device)


        anchor = pc.get_anchor[visible_mask]
        feat = pc.get_anchor_feat[visible_mask]
        level = pc.get_level[visible_mask]
        # Hash-based LOD: clamp level to available hash layers
        if hasattr(pc, "use_hash_encoding") and pc.use_hash_encoding and getattr(pc, "hash_levels", None) is not None and int(getattr(pc, "hash_levels", 0)) > 0:
            level = torch.clamp(level, max=int(pc.hash_levels) - 1)

        use_hash_interpolation = getattr(pc, "use_hash_interpolation", True)

        # Query the AGD hash grid when enabled.
        if getattr(pc, "use_hash_feat_multi_level", False):
            hash_feat = pc.query_hash_feat_multi_level(anchor, use_interpolation=use_hash_interpolation)
            if hash_feat is not None:
                feat = hash_feat
        elif getattr(pc, "use_hash_feat_single_level", False):
            hash_feat = pc.query_hash_feat_single_level(anchor, level, use_interpolation=use_hash_interpolation)
            if hash_feat is not None:
                feat = hash_feat

        grid_offsets = pc._offset[visible_mask]
        grid_scaling = pc.get_scaling[visible_mask]


        camera_center = viewpoint_camera.camera_center
        ob_view = anchor - camera_center
        ob_dist = torch.norm(ob_view, dim=1, keepdim=True)
        ob_view = ob_view / ob_dist



        feature_combinations = {}


        if pc.use_feat_bank:
            if pc.add_level:
                cat_view = torch.cat([ob_view, level], dim=1)
            else:
                cat_view = ob_view

            bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1)


            feat_unsqueezed = feat.unsqueeze(dim=-1)

            feat = feat_unsqueezed[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
                  feat_unsqueezed[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
                  feat_unsqueezed[:,::1, :1]*bank_weight[:,:,2:]
            feat = feat.squeeze(dim=-1)


        if pc.add_level:
            cat_local_view = torch.cat([feat, ob_view, ob_dist, level], dim=1)
            cat_local_view_wodist = torch.cat([feat, ob_view, level], dim=1)
        else:
            cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)
            cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)

        feature_combinations['cat_local_view'] = cat_local_view
        feature_combinations['cat_local_view_wodist'] = cat_local_view_wodist


        if pc.appearance_dim > 0:
            if is_training or ape_code < 0:
                camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=device) * viewpoint_camera.uid
            else:
                camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=device) * ape_code[0]

            appearance = pc.get_appearance(camera_indicies)
            feature_combinations['appearance'] = appearance


            if pc.add_color_dist:
                feature_combinations['color_input'] = torch.cat([cat_local_view, appearance], dim=1)
            else:
                feature_combinations['color_input'] = torch.cat([cat_local_view_wodist, appearance], dim=1)


        neural_opacity_input = cat_local_view if pc.add_opacity_dist else cat_local_view_wodist
        neural_opacity = pc.get_opacity_mlp(neural_opacity_input)


        if pc.dist2level == "progressive":
            prog = pc._prog_ratio[visible_mask]
            transition_mask = pc.transition_mask[visible_mask]
            prog[~transition_mask] = 1.0
            neural_opacity = neural_opacity * prog

        # Apply LOD opacity bias from AGD.
        neural_opacity = self.apply_lod_opacity_adjustment(neural_opacity, pc, visible_mask)


        neural_opacity = neural_opacity.reshape([-1, 1])
        mask = (neural_opacity > 0.0).view(-1)


        opacity = neural_opacity[mask]



        if pc.appearance_dim > 0:
            color_input = feature_combinations['color_input']
        else:
            color_input = cat_local_view if pc.add_color_dist else cat_local_view_wodist

        color = pc.get_color_mlp(color_input).reshape([anchor.shape[0]*pc.n_offsets, 3])


        cov_input = cat_local_view if pc.add_cov_dist else cat_local_view_wodist
        scale_rot = pc.get_cov_mlp(cov_input).reshape([anchor.shape[0]*pc.n_offsets, 7])


        offsets = grid_offsets.view([-1, 3])


        concatenated = torch.cat([grid_scaling, anchor], dim=-1)

        concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)

        concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)

        masked = concatenated_all[mask]

        scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)

        if torch.isnan(scaling_repeat).any():
            scaling_repeat = torch.nan_to_num(scaling_repeat, nan=1e-3)
        scaling_repeat = torch.clamp(scaling_repeat, min=1e-3)


        scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3])
        scaling = torch.clamp(scaling, min=1e-3)
        rot = pc.rotation_activation(scale_rot[:,3:7])


        offsets = offsets * scaling_repeat[:,:3]
        xyz = repeat_anchor + offsets


        rotations_mat = quaternion_to_matrix(rot)
        min_scales = torch.argmin(scaling, dim=-1)
        indices = torch.arange(min_scales.shape[0], device=device)
        normals = rotations_mat[indices, :, min_scales]


        dir_pp = xyz - camera_center.repeat(xyz.shape[0], 1)
        dir_pp_norm = torch.norm(dir_pp, dim=1, keepdim=True)
        dir_pp_normalized = dir_pp / (dir_pp_norm + 1e-8)
        dotprod = torch.sum(-dir_pp_normalized * normals, dim=1, keepdim=True)
        normals = torch.where(dotprod >= 0, normals, -normals)


        semantic = pc._semantic[visible_mask]
        concatenated_semantic = repeat(semantic, 'n (c) -> (n k) (c)', k=pc.n_offsets)
        semantics = concatenated_semantic[mask]

        # Color is precomputed by the MLP, so SH features are not passed.
        shs_features = None

        if is_training:
            return xyz, color, opacity, scaling, rot, neural_opacity, mask, normals, semantics, shs_features
        else:
            return xyz, color, opacity, scaling, rot, normals, semantics, shs_features

    def generate_neural_gaussians_obj(self, pipe, viewpoint_camera, pc, visible_mask=None, is_training=False, ape_code=-1, stage="fine"):
        """
        Generate Gaussian attributes for a rigid actor.
        """
        device = pc.get_anchor.device


        if visible_mask is None:
            visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device=device)


        obj_rot = quaternion_to_matrix(pc.obj_rots)


        anchor_obj = torch.einsum('bij, bj -> bi', obj_rot, pc.get_anchor) + pc.obj_trans


        anchor = anchor_obj[visible_mask]
        feat = pc.get_anchor_feat[visible_mask]
        level = pc.get_level[visible_mask]
        if hasattr(pc, "use_hash_encoding") and pc.use_hash_encoding and getattr(pc, "hash_levels", None) is not None and int(getattr(pc, "hash_levels", 0)) > 0:
            level = torch.clamp(level, max=int(pc.hash_levels) - 1)
        grid_offsets = pc._offset[visible_mask]
        grid_scaling = pc.get_scaling[visible_mask]


        camera_center = viewpoint_camera.camera_center
        ob_view = anchor - camera_center
        ob_dist = torch.norm(ob_view, dim=1, keepdim=True)
        ob_view = ob_view / ob_dist


        feature_combinations = {}


        if pc.use_feat_bank:
            if pc.add_level:
                cat_view = torch.cat([ob_view, level], dim=1)
            else:
                cat_view = ob_view

            bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1)


            feat_unsqueezed = feat.unsqueeze(dim=-1)

            feat = feat_unsqueezed[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
                  feat_unsqueezed[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
                  feat_unsqueezed[:,::1, :1]*bank_weight[:,:,2:]
            feat = feat.squeeze(dim=-1)


        if pc.add_level:
            cat_local_view = torch.cat([feat, ob_view, ob_dist, level], dim=1)
            cat_local_view_wodist = torch.cat([feat, ob_view, level], dim=1)
        else:
            cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)
            cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)

        feature_combinations['cat_local_view'] = cat_local_view
        feature_combinations['cat_local_view_wodist'] = cat_local_view_wodist


        if pc.appearance_dim > 0:
            if is_training or ape_code < 0:
                camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=device) * viewpoint_camera.uid
            else:
                camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=device) * ape_code[0]

            appearance = pc.get_appearance(camera_indicies)
            feature_combinations['appearance'] = appearance


            if pc.add_color_dist:
                feature_combinations['color_input'] = torch.cat([cat_local_view, appearance], dim=1)
            else:
                feature_combinations['color_input'] = torch.cat([cat_local_view_wodist, appearance], dim=1)


        neural_opacity_input = cat_local_view if pc.add_opacity_dist else cat_local_view_wodist
        neural_opacity = pc.get_opacity_mlp(neural_opacity_input)


        if pc.dist2level == "progressive":
            prog = pc._prog_ratio[visible_mask]
            transition_mask = pc.transition_mask[visible_mask]
            prog[~transition_mask] = 1.0
            neural_opacity = neural_opacity * prog

        # Apply LOD opacity bias from AGD.
        neural_opacity = self.apply_lod_opacity_adjustment(neural_opacity, pc, visible_mask)


        neural_opacity = neural_opacity.reshape([-1, 1])
        mask = (neural_opacity > 0.0).view(-1)


        opacity = neural_opacity[mask]



        if pc.appearance_dim > 0:
            color_input = feature_combinations['color_input']
        else:
            color_input = cat_local_view if pc.add_color_dist else cat_local_view_wodist

        color = pc.get_color_mlp(color_input).reshape([anchor.shape[0]*pc.n_offsets, 3])


        cov_input = cat_local_view if pc.add_cov_dist else cat_local_view_wodist
        scale_rot = pc.get_cov_mlp(cov_input).reshape([anchor.shape[0]*pc.n_offsets, 7])


        offsets = grid_offsets.view([-1, 3])


        concatenated = torch.cat([grid_scaling, anchor], dim=-1)

        concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)

        concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)

        masked = concatenated_all[mask]

        scaling_repeat, repeat_anchor, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)

        if torch.isnan(scaling_repeat).any():
            scaling_repeat = torch.nan_to_num(scaling_repeat, nan=1e-3)
        scaling_repeat = torch.clamp(scaling_repeat, min=1e-3)


        scaling = scaling_repeat[:,3:] * torch.sigmoid(scale_rot[:,:3])
        scaling = torch.clamp(scaling, min=1e-3)
        rot = pc.rotation_activation(scale_rot[:,3:7])


        offsets = offsets * scaling_repeat[:,:3]
        xyz = repeat_anchor + offsets

        rotations_mat = quaternion_to_matrix(rot)
        min_scales = torch.argmin(scaling, dim=-1)
        indices = torch.arange(min_scales.shape[0], device=device)
        normals = rotations_mat[indices, :, min_scales]


        dir_pp = xyz - camera_center.expand(xyz.shape[0], -1)
        dir_pp_norm = torch.norm(dir_pp, dim=1, keepdim=True)
        dir_pp_normalized = dir_pp / (dir_pp_norm + 1e-8)
        dotprod = torch.sum(-dir_pp_normalized * normals, dim=1, keepdim=True)
        normals = torch.where(dotprod >= 0, normals, -normals)


        semantic = pc._semantic[visible_mask]
        concatenated_semantic = repeat(semantic, 'n (c) -> (n k) (c)', k=pc.n_offsets)
        semantics = concatenated_semantic[mask]

        # Color is precomputed by the MLP, so SH features are not passed.
        shs_features = None



        if is_training:
            return xyz, color, opacity, scaling, rot, neural_opacity, mask, normals, semantics, shs_features
        else:
            return xyz, color, opacity, scaling, rot, normals, semantics, shs_features

    def generate_neural_gaussians_non_rigid(self, pipe, viewpoint_camera, pc, visible_mask=None, is_training=False, ape_code=-1, stage="fine", total_frames=50, iteration=0, deform=None, is_infer=False, hyper=None):
        """
        Generate Gaussian attributes for an actor with AGD non-rigid deformation.
        """
        import os
        # Enable one-shot diagnostic logs with --debug_from and DRIVESPLAT_DEBUG_LOG=1.
        debug_enabled = bool(getattr(pipe, "debug", False)) and (os.environ.get("DRIVESPLAT_DEBUG_LOG", "0") == "1")


        device = pc.get_anchor.device


        if visible_mask is None:
            visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device=device)
        if pc.obj_class == 'pedestrian' and torch.sum(visible_mask) <= 0:
            visible_mask = torch.ones(pc.get_anchor.shape[0], dtype=torch.bool, device=device)


        # Keep anchors in canonical actor coordinates for AGD, then transform to world space.
        obj_rots_input = pc.obj_rots
        obj_rot = quaternion_to_matrix(obj_rots_input)

        if not torch.is_grad_enabled():
            obj_rot = obj_rot
            obj_trans_input = pc.obj_trans
        else:
            obj_trans_input = pc.obj_trans

        anchor_canonical = pc.get_anchor[visible_mask]

        anchor_obj = torch.einsum('bij, bj -> bi', obj_rot, pc.get_anchor) + obj_trans_input
        anchor = anchor_obj[visible_mask]

        feat = pc.get_anchor_feat[visible_mask]
        level = pc.get_level[visible_mask]
        if hasattr(pc, "use_hash_encoding") and pc.use_hash_encoding and getattr(pc, "hash_levels", None) is not None and int(getattr(pc, "hash_levels", 0)) > 0:
            level = torch.clamp(level, max=int(pc.hash_levels) - 1)

        # Match the rigid actor path: dynamic/non-rigid actors must use the
        # trained hash features for opacity/color/covariance and AGD input.
        use_hash_interpolation = getattr(pc, "use_hash_interpolation", True)
        if getattr(pc, "use_hash_feat_multi_level", False):
            hash_feat = pc.query_hash_feat_multi_level(anchor_canonical, use_interpolation=use_hash_interpolation)
            if hash_feat is not None:
                feat = hash_feat
        elif getattr(pc, "use_hash_feat_single_level", False):
            hash_feat = pc.query_hash_feat_single_level(anchor_canonical, level, use_interpolation=use_hash_interpolation)
            if hash_feat is not None:
                feat = hash_feat

        grid_offsets = pc._offset[visible_mask]
        grid_scaling = pc.get_scaling[visible_mask]

        camera_center = viewpoint_camera.camera_center
        ob_view = anchor - camera_center
        ob_dist = torch.norm(ob_view, dim=1, keepdim=True)
        ob_view = ob_view / ob_dist


        feature_combinations = {}


        if pc.use_feat_bank:
            cat_view = torch.cat([ob_view, level], dim=1) if pc.add_level else ob_view

            bank_weight = pc.get_featurebank_mlp(cat_view).unsqueeze(dim=1)


            feat_unsqueezed = feat.unsqueeze(dim=-1)
            feat = feat_unsqueezed[:,::4, :1].repeat([1,4,1])*bank_weight[:,:,:1] + \
                  feat_unsqueezed[:,::2, :1].repeat([1,2,1])*bank_weight[:,:,1:2] + \
                  feat_unsqueezed[:,::1, :1]*bank_weight[:,:,2:]
            feat = feat.squeeze(dim=-1)


        if pc.add_level:
            cat_local_view = torch.cat([feat, ob_view, ob_dist, level], dim=1)
            cat_local_view_wodist = torch.cat([feat, ob_view, level], dim=1)
        else:
            cat_local_view = torch.cat([feat, ob_view, ob_dist], dim=1)
            cat_local_view_wodist = torch.cat([feat, ob_view], dim=1)

        feature_combinations['cat_local_view'] = cat_local_view
        feature_combinations['cat_local_view_wodist'] = cat_local_view_wodist


        if pc.appearance_dim > 0:
            camera_indicies = torch.ones_like(cat_local_view[:,0], dtype=torch.long, device=device)
            if is_training or ape_code < 0:
                camera_indicies = camera_indicies * viewpoint_camera.uid
            else:
                camera_indicies = camera_indicies * ape_code[0]

            appearance = pc.get_appearance(camera_indicies)
            feature_combinations['appearance'] = appearance


            if pc.add_color_dist:
                feature_combinations['color_input'] = torch.cat([cat_local_view, appearance], dim=1)
            else:
                feature_combinations['color_input'] = torch.cat([cat_local_view_wodist, appearance], dim=1)


        neural_opacity_input = cat_local_view if pc.add_opacity_dist else cat_local_view_wodist
        neural_opacity_raw = pc.get_opacity_mlp(neural_opacity_input)  # MLP tanh output in [-1, 1].


        if pc.dist2level == "progressive":
            prog = pc._prog_ratio[visible_mask]
            transition_mask = pc.transition_mask[visible_mask]
            prog[~transition_mask] = 1.0
            neural_opacity_raw = neural_opacity_raw * prog

        # Map tanh opacity logits to [0, 1] for stable non-rigid actor masks.
        neural_opacity = (neural_opacity_raw + 1.0) * 0.5


        neural_opacity = neural_opacity.reshape([-1, 1])

        # Warm up the opacity threshold together with non-rigid deformation.
        opacity_thresh_target = float(getattr(hyper, "opacity_thresh", 0.001)) if hyper is not None else 0.001
        if is_infer:
            opacity_thresh = opacity_thresh_target
        else:
            if 'non_rigid_start_iter' in locals() and iteration < non_rigid_start_iter:
                opacity_thresh = 0.0
            elif 'non_rigid_start_iter' in locals() and 'non_rigid_warmup_iter' in locals() and non_rigid_warmup_iter > 0:
                prog = max(0.0, min(1.0, (iteration - non_rigid_start_iter) / float(non_rigid_warmup_iter)))
                opacity_thresh = opacity_thresh_target * prog
            else:
                opacity_thresh = opacity_thresh_target
        mask = (neural_opacity > opacity_thresh).view(-1)
        mask_init = mask.clone()

        # Track which anchor generated each selected Gaussian.
        anchor_indices = torch.arange(anchor_canonical.shape[0], device=device)
        anchor_indices_repeat = repeat(anchor_indices, 'n -> (n k)', k=pc.n_offsets)[mask]  # (num_gaussians,)


        opacity = neural_opacity[mask]


        if pc.appearance_dim > 0:
            color_input = feature_combinations['color_input']
        else:
            color_input = cat_local_view if pc.add_color_dist else cat_local_view_wodist

        color_temp = pc.get_color_mlp(color_input)
        color_before_deform = color_temp.reshape([anchor.shape[0]*pc.n_offsets, 3])

        color = color_before_deform


        cov_input = cat_local_view if pc.add_cov_dist else cat_local_view_wodist
        scale_rot = pc.get_cov_mlp(cov_input).reshape([anchor.shape[0]*pc.n_offsets, 7])


        offsets = grid_offsets.view([-1, 3])


        concatenated = torch.cat([grid_scaling, anchor_canonical], dim=-1)
        concatenated_repeated = repeat(concatenated, 'n (c) -> (n k) (c)', k=pc.n_offsets)
        concatenated_all = torch.cat([concatenated_repeated, color, scale_rot, offsets], dim=-1)
        masked = concatenated_all[mask]
        scaling_repeat, repeat_anchor_canonical, color, scale_rot, offsets = masked.split([6, 3, 3, 7, 3], dim=-1)

        if torch.isnan(scaling_repeat).any():
            scaling_repeat = torch.nan_to_num(scaling_repeat, nan=1e-3)
        scaling_repeat = torch.clamp(scaling_repeat, min=1e-3)


        scaling = scaling_repeat[:, 3:] * torch.sigmoid(scale_rot[:, :3])
        scaling = torch.clamp(scaling, min=1e-3)
        rot = pc.rotation_activation(scale_rot[:, 3:7])

        offsets = offsets * scaling_repeat[:, :3]
        xyz_canonical = repeat_anchor_canonical + offsets  # Canonical space

        # Expand per-anchor LOD tree levels to per-Gaussian levels.
        level_anchor = level
        if level_anchor.dim() == 2:
            level_anchor = level_anchor.squeeze(-1)
        level_repeat = repeat(level_anchor, 'n -> (n k)', k=pc.n_offsets)
        level_repeat = level_repeat[mask]  # (num_gaussians,)


        semantic = pc._semantic[visible_mask]
        concatenated_semantic = repeat(semantic, 'n (c) -> (n k) (c)', k=pc.n_offsets)
        semantics = concatenated_semantic[mask]

        shs = None

        # AGD non-rigid deformation schedule.
        if hyper is not None:
            non_rigid_start_iter = getattr(hyper, 'non_rigid_start_iter', 3000)
            non_rigid_warmup_iter = getattr(hyper, 'non_rigid_warmup_iter', 2000)
        else:
            non_rigid_start_iter = getattr(pc, 'non_rigid_start_iter', 3000) if hasattr(pc, 'non_rigid_start_iter') else 3000
            non_rigid_warmup_iter = getattr(pc, 'non_rigid_warmup_iter', 2000) if hasattr(pc, 'non_rigid_warmup_iter') else 2000

        if is_infer:
            non_rigid_weight = 1.0
        elif iteration < non_rigid_start_iter:
            non_rigid_weight = 0.0
        elif iteration < non_rigid_start_iter + non_rigid_warmup_iter:

            progress = (iteration - non_rigid_start_iter) / non_rigid_warmup_iter
            non_rigid_weight = progress ** 0.5
        else:
            non_rigid_weight = 1.0

        enable_non_rigid = (non_rigid_weight > 0.0)


        use_deform = stage == "fine" and enable_non_rigid and hasattr(pc, "deform")



        if use_deform:
            N = xyz_canonical.shape[0]

            fid = viewpoint_camera.uid
            if hasattr(viewpoint_camera, "meta"):
                fid = viewpoint_camera.meta.get("frame_idx", fid)

            #  [0, 1]:actorstart_frameend_frame
            if hasattr(pc, 'start_frame') and hasattr(pc, 'end_frame'):
                denom = max(1, (pc.end_frame - pc.start_frame))
                t_norm = (fid - pc.start_frame) / denom
            else:
                # fallback: total_frames
                t_norm = fid / total_frames if total_frames > 0 else 0.0


            t_norm = torch.clamp(torch.tensor(t_norm, dtype=torch.float32, device='cuda'), 0.0, 1.0)
            if N > 0:
                time_input = t_norm.unsqueeze(0).expand(N, -1)
            else:
                time_input = torch.empty((0, 1), dtype=torch.float32, device='cuda')

            if debug_enabled and not hasattr(pc, '_debug_time_printed'):
                pc._debug_time_printed = True
                print(f"[Time] Actor {getattr(pc, 'obj_id', 'unknown')}: fid={fid}, start_frame={getattr(pc, 'start_frame', 'N/A')}, end_frame={getattr(pc, 'end_frame', 'N/A')}, t_norm={t_norm.item():.4f}")


            if debug_enabled and is_infer and not hasattr(pc, '_debug_infer_time_printed'):
                pc._debug_infer_time_printed = True
                print(f"[INFER] Time info: fid={fid}, t_norm={t_norm.item():.4f}, time_input.shape={time_input.shape}")



            # AGD predicts canonical-space residuals from the anchor state.
            xyz_input = xyz_canonical

            d_xyz, d_rotation, d_scaling, d_color_residual = pc.deform.step(
                xyz_input,
                time_input,
                level=level_repeat,
                use_euler=False,
                dt=None,
                anchor=anchor_canonical,
                offsets=offsets,
                scaling=scaling,
                rotation=rot,
                non_rigid_weight=non_rigid_weight,
                anchor_feat=feat,
                anchor_to_gaussian_idx=anchor_indices_repeat,
            )
            original_mask_before_deform = mask.clone()
            num_gaussians_before_deform = original_mask_before_deform.sum().item()

            xyz_deformed = xyz_canonical + d_xyz  # canonical space


            if debug_enabled and not hasattr(pc, '_debug_deform_disappear'):
                pc._debug_deform_disappear = True
                with torch.no_grad():
                    xyz_canonical_norm = torch.norm(xyz_canonical, dim=-1)
                    d_xyz_norm = torch.norm(d_xyz, dim=-1)
                    xyz_deformed_norm = torch.norm(xyz_deformed, dim=-1)
                    print(f"\n[NonRigid Debug]")
                    print(f"  xyz_canonical: mean_norm={xyz_canonical_norm.mean().item():.3f}, max_norm={xyz_canonical_norm.max().item():.3f}")
                    print(f"  d_xyz: mean_norm={d_xyz_norm.mean().item():.6f}, max_norm={d_xyz_norm.max().item():.6f}")
                    print(f"  xyz_deformed: mean_norm={xyz_deformed_norm.mean().item():.3f}, max_norm={xyz_deformed_norm.max().item():.3f}")
                    print(f"  d_xyz / xyz_canonical ratio: {(d_xyz_norm.mean() / xyz_canonical_norm.mean()).item():.6f}")

                    num_large_deform = (d_xyz_norm > 1.0).sum().item()
                    num_extreme_deform = (d_xyz_norm > 10.0).sum().item()
                    print(f"  Large deform (>1.0m): {num_large_deform}/{d_xyz.shape[0]}")
                    print(f"  Extreme deform (>10.0m): {num_extreme_deform}/{d_xyz.shape[0]}")
                    print(f"  xyz_deformed range: [{xyz_deformed.min().item():.3f}, {xyz_deformed.max().item():.3f}]\n")
            obj_rot_masked = obj_rot[visible_mask]  # (N, 3, 3)
            obj_trans_masked = obj_trans_input[visible_mask]  # (N, 3)

            obj_rot_selected = obj_rot_masked[anchor_indices_repeat]  # (num_gaussians, 3, 3)
            obj_trans_selected = obj_trans_masked[anchor_indices_repeat]  # (num_gaussians, 3)

            if obj_rot_selected.shape[0] != xyz_deformed.shape[0]:
                raise ValueError(
                    f"obj_rot_selected.shape[0]={obj_rot_selected.shape[0]} != xyz_deformed.shape[0]={xyz_deformed.shape[0]}, "
                    f"anchor_indices_repeat.shape[0]={anchor_indices_repeat.shape[0]}, mask.sum()={mask.sum().item()}. "
                    f"This will cause xyz shape mismatch!"
                )
            original_mask_before_deform = mask.clone()
            num_gaussians_before_deform = original_mask_before_deform.sum().item()

            xyz = torch.einsum('bij, bj -> bi', obj_rot_selected, xyz_deformed) + obj_trans_selected

            if debug_enabled and not hasattr(pc, '_debug_world_xyz'):
                pc._debug_world_xyz = True
                with torch.no_grad():
                    xyz_norm = torch.norm(xyz, dim=-1)
                    obj_trans_norm = torch.norm(obj_trans_selected, dim=-1)
                    print(f"\n[World Coord Debug]")
                    print(f"  xyz (world): mean={xyz.mean(dim=0).cpu().numpy()}, std={xyz.std(dim=0).cpu().numpy()}")
                    print(f"  xyz range: [{xyz.min().item():.3f}, {xyz.max().item():.3f}]")
                    print(f"  xyz norm: mean={xyz_norm.mean().item():.3f}, max={xyz_norm.max().item():.3f}")
                    print(f"  obj_trans: mean={obj_trans_selected.mean(dim=0).cpu().numpy()}")
                    print(f"  obj_trans norm: mean={obj_trans_norm.mean().item():.3f}")

                    num_far = (xyz_norm > 100.0).sum().item()
                    num_extreme_far = (xyz_norm > 1000.0).sum().item()
                    print(f"  Far from origin (>100m): {num_far}/{xyz.shape[0]}")
                    print(f"  Extremely far (>1000m): {num_extreme_far}/{xyz.shape[0]}\n")
            if xyz.shape[0] != num_gaussians_before_deform:
                raise ValueError(f"xyz.shape[0]={xyz.shape[0]} != num_gaussians_before_deform={num_gaussians_before_deform}, mask.sum()={original_mask_before_deform.sum().item()}. This will cause gradient shape mismatch! xyz shape must match original mask. obj_rot_selected.shape[0]={obj_rot_selected.shape[0]}, xyz_deformed.shape[0]={xyz_deformed.shape[0]}")

            num_gaussians = num_gaussians_before_deform

            ob_view_new = xyz - camera_center.expand(xyz.shape[0], -1)
            ob_dist_new = torch.norm(ob_view_new, dim=1, keepdim=True)
            ob_view_new = ob_view_new / ob_dist_new

            feat_per_gaussian = feat[anchor_indices_repeat]
            if level_anchor.dim() == 2:
                level_anchor_flat = level_anchor.squeeze(-1)
            else:
                level_anchor_flat = level_anchor
            level_per_gaussian = level_anchor_flat[anchor_indices_repeat]

            if feat_per_gaussian.shape[0] != num_gaussians:
                if feat_per_gaussian.shape[0] > num_gaussians:
                    feat_per_gaussian = feat_per_gaussian[:num_gaussians]
                    level_per_gaussian = level_per_gaussian[:num_gaussians]
                else:
                    raise ValueError(f"feat_per_gaussian.shape[0]={feat_per_gaussian.shape[0]} < num_gaussians={num_gaussians}, mask.sum()={mask.sum()}, xyz.shape[0]={xyz.shape[0]}")

            if pc.use_feat_bank:
                cat_view_new = torch.cat([ob_view_new, level_per_gaussian.unsqueeze(-1)], dim=1) if pc.add_level else ob_view_new

                bank_weight_new = pc.get_featurebank_mlp(cat_view_new).unsqueeze(dim=1)



                feat_unsqueezed_new = feat_per_gaussian.unsqueeze(dim=-1)
                feat_per_gaussian = feat_unsqueezed_new[:,::4, :1].repeat([1,4,1])*bank_weight_new[:,:,:1] + \
                      feat_unsqueezed_new[:,::2, :1].repeat([1,2,1])*bank_weight_new[:,:,1:2] + \
                      feat_unsqueezed_new[:,::1, :1]*bank_weight_new[:,:,2:]
                feat_per_gaussian = feat_per_gaussian.squeeze(dim=-1)

                del feat_unsqueezed_new, bank_weight_new, cat_view_new



            if pc.add_level:
                cat_local_view_new = torch.cat([feat_per_gaussian, ob_view_new, ob_dist_new, level_per_gaussian.unsqueeze(-1)], dim=1)
                cat_local_view_wodist_new = torch.cat([feat_per_gaussian, ob_view_new, level_per_gaussian.unsqueeze(-1)], dim=1)
            else:
                cat_local_view_new = torch.cat([feat_per_gaussian, ob_view_new, ob_dist_new], dim=1)
                cat_local_view_wodist_new = torch.cat([feat_per_gaussian, ob_view_new], dim=1)

            neural_opacity_input_new = cat_local_view_new if pc.add_opacity_dist else cat_local_view_wodist_new

            neural_opacity_raw_new = pc.get_opacity_mlp(neural_opacity_input_new)  # MLP tanh output in [-1, 1].
            del neural_opacity_input_new


            if pc.dist2level == "progressive":
                if hasattr(pc, '_prog_ratio'):
                    prog_anchor = pc._prog_ratio[visible_mask]
                    prog_per_gaussian = prog_anchor[anchor_indices_repeat]
                    if hasattr(pc, 'transition_mask'):
                        transition_mask_anchor = pc.transition_mask[visible_mask]
                        transition_mask_per_gaussian = transition_mask_anchor[anchor_indices_repeat]
                        prog_per_gaussian = prog_per_gaussian.clone()  # in-place
                        prog_per_gaussian[~transition_mask_per_gaussian] = 1.0
                        del transition_mask_per_gaussian
                    neural_opacity_raw_new = neural_opacity_raw_new * prog_per_gaussian.unsqueeze(-1)

                    del prog_per_gaussian

            neural_opacity_new = (neural_opacity_raw_new + 1.0) * 0.5
            neural_opacity_new = neural_opacity_new.reshape([-1, 1])
            del neural_opacity_raw_new

            if neural_opacity_new.shape[0] != num_gaussians:
                if neural_opacity_new.shape[0] > num_gaussians:
                    neural_opacity_new = neural_opacity_new[:num_gaussians]
                else:
                    raise ValueError(f"neural_opacity_new.shape[0]={neural_opacity_new.shape[0]} < num_gaussians={num_gaussians}, xyz.shape[0]={xyz.shape[0]}")

            opacity = neural_opacity_new

            if debug_enabled and not hasattr(pc, '_debug_opacity_check'):
                pc._debug_opacity_check = True
                with torch.no_grad():
                    opacity_mean = opacity.mean().item()
                    opacity_min = opacity.min().item()
                    opacity_max = opacity.max().item()
                    num_high_opacity = (opacity > 0.5).sum().item()
                    num_low_opacity = (opacity < 0.01).sum().item()
                    print(f"\n[Opacity Debug] Opacity after non-rigid:")
                    print(f"  - mean: {opacity_mean:.4f}, min: {opacity_min:.4f}, max: {opacity_max:.4f}")
                    print(f"  - high (>0.5): {num_high_opacity}/{opacity.shape[0]}")
                    print(f"  - low (<0.01): {num_low_opacity}/{opacity.shape[0]}\n")

            if pc.appearance_dim > 0:
                if 'cat_local_view_new' in locals() and cat_local_view_new.numel() > 0:
                    camera_indicies_new = torch.ones_like(cat_local_view_new[:,0], dtype=torch.long, device=device)
                elif 'cat_local_view_wodist_new' in locals() and cat_local_view_wodist_new.numel() > 0:
                    camera_indicies_new = torch.ones_like(cat_local_view_wodist_new[:,0], dtype=torch.long, device=device)
                else:
                    camera_indicies_new = torch.ones(num_gaussians, dtype=torch.long, device=device)

                if is_training or ape_code < 0:
                    camera_indicies_new = camera_indicies_new * viewpoint_camera.uid
                else:
                    camera_indicies_new = camera_indicies_new * ape_code[0]
                appearance_new = pc.get_appearance(camera_indicies_new)

                if pc.add_color_dist:
                    if 'cat_local_view_new' not in locals() or cat_local_view_new.numel() == 0:
                        if pc.add_level:
                            cat_local_view_new = torch.cat([feat_per_gaussian, ob_view_new, ob_dist_new, level_per_gaussian.unsqueeze(-1)], dim=1)
                        else:
                            cat_local_view_new = torch.cat([feat_per_gaussian, ob_view_new, ob_dist_new], dim=1)
                    color_input_new = torch.cat([cat_local_view_new, appearance_new], dim=1)
                    del cat_local_view_new
                else:
                    if 'cat_local_view_wodist_new' not in locals() or cat_local_view_wodist_new.numel() == 0:
                        if pc.add_level:
                            cat_local_view_wodist_new = torch.cat([feat_per_gaussian, ob_view_new, level_per_gaussian.unsqueeze(-1)], dim=1)
                        else:
                            cat_local_view_wodist_new = torch.cat([feat_per_gaussian, ob_view_new], dim=1)
                    color_input_new = torch.cat([cat_local_view_wodist_new, appearance_new], dim=1)
                    del cat_local_view_wodist_new

                del appearance_new, camera_indicies_new
            else:
                if pc.add_color_dist:
                    if 'cat_local_view_new' not in locals() or cat_local_view_new.numel() == 0:
                        if pc.add_level:
                            cat_local_view_new = torch.cat([feat_per_gaussian, ob_view_new, ob_dist_new, level_per_gaussian.unsqueeze(-1)], dim=1)
                        else:
                            cat_local_view_new = torch.cat([feat_per_gaussian, ob_view_new, ob_dist_new], dim=1)
                    color_input_new = cat_local_view_new
                    del cat_local_view_new
                else:
                    if 'cat_local_view_wodist_new' not in locals() or cat_local_view_wodist_new.numel() == 0:
                        if pc.add_level:
                            cat_local_view_wodist_new = torch.cat([feat_per_gaussian, ob_view_new, level_per_gaussian.unsqueeze(-1)], dim=1)
                        else:
                            cat_local_view_wodist_new = torch.cat([feat_per_gaussian, ob_view_new], dim=1)
                    color_input_new = cat_local_view_wodist_new
                    del cat_local_view_wodist_new

            color_temp_new = pc.get_color_mlp(color_input_new)
            del color_input_new
            color = color_temp_new.reshape([-1, 3])
            del color_temp_new
            color = torch.clamp(color, 0.0, 1.0)

            if color.shape[0] != num_gaussians:
                if color.shape[0] > num_gaussians:
                    color = color[:num_gaussians]
                else:
                    raise ValueError(f"color.shape[0]={color.shape[0]} < num_gaussians={num_gaussians}, xyz.shape[0]={xyz.shape[0]}")

            del neural_opacity_new
            if 'feat_per_gaussian' in locals():
                del feat_per_gaussian
            if 'ob_view_new' in locals():
                del ob_view_new
            if 'ob_dist_new' in locals():
                del ob_dist_new
            if 'level_per_gaussian' in locals():
                del level_per_gaussian
            scaling_before = scaling.clone()

            # Delay rotation/scale residuals until the non-rigid position field is warm.
            attr_start_iter = int(getattr(hyper, "non_rigid_attr_start_iter", 2000)) if hyper is not None else 2000
            if (not is_infer) and iteration < attr_start_iter:
                d_rotation = torch.zeros_like(d_rotation)
                d_scaling = torch.zeros_like(d_scaling)

            scale_factor = float(getattr(hyper, "non_rigid_scale_factor", 0.05)) if hyper is not None else 0.05
            scaling = scaling * torch.exp(d_scaling * scale_factor)

            if debug_enabled and not hasattr(pc, '_debug_scaling_check'):
                pc._debug_scaling_check = True
                with torch.no_grad():
                    scaling_before_clamp_mean = scaling.mean().item()
                    scaling_before_clamp_min = scaling.min().item()
                    scaling_before_clamp_max = scaling.max().item()
                    num_too_small = (scaling < 1e-3).any(dim=-1).sum().item()
                    num_negative = (scaling < 0).any(dim=-1).sum().item()
                    print(f"\n[Scaling Debug] After exp update:")
                    print(f"  - mean: {scaling_before_clamp_mean:.6f}, min: {scaling_before_clamp_min:.6f}, max: {scaling_before_clamp_max:.6f}")
                    print(f"  - too small (<1e-3): {num_too_small}/{scaling.shape[0]}")
                    print(f"  - negative (<0): {num_negative}/{scaling.shape[0]}\n")


            scaling = torch.clamp(scaling, min=1e-3, max=1.0)
            with torch.no_grad():
                if debug_enabled and not hasattr(pc, '_debug_scaling_clamp_printed'):
                    pc._debug_scaling_clamp_printed = True
                    num_clamped_min = (scaling == 1e-4).any(dim=-1).sum().item()
                    num_clamped_max = (scaling == 1.0).any(dim=-1).sum().item()
                    scaling_mean = scaling.mean().item()
                    scaling_min = scaling.min().item()
                    scaling_max = scaling.max().item()
                    d_scaling_mean = d_scaling.abs().mean().item()
                    print(f"[DEBUG] Scaling stats: mean={scaling_mean:.6f}, min={scaling_min:.6f}, max={scaling_max:.6f}")
                    print(f"[DEBUG] d_scaling mean abs={d_scaling_mean:.6f}, clamped_min={num_clamped_min}, clamped_max={num_clamped_max}")
                    print(f"[DEBUG] scaling_before mean={scaling_before.mean().item():.6f}, scaling_after mean={scaling_mean:.6f}")

            rot = rot + d_rotation
            rot_norm = torch.norm(rot, dim=-1, keepdim=True)
            rot = rot / (rot_norm + 1e-8)

            if debug_enabled and is_infer and not hasattr(pc, '_debug_infer_final_printed'):
                pc._debug_infer_final_printed = True
                with torch.no_grad():

                    if xyz.numel() > 0 and xyz_canonical.numel() > 0:
                        xyz_diff = torch.norm(xyz - xyz_canonical, dim=-1).mean()
                        xyz_diff_max = torch.norm(xyz - xyz_canonical, dim=-1).max()
                        xyz_min = xyz.min().item()
                        xyz_max = xyz.max().item()
                    else:
                        xyz_diff = torch.tensor(0.0, device=xyz.device if xyz.numel() > 0 else xyz_canonical.device)
                        xyz_diff_max = torch.tensor(0.0, device=xyz.device if xyz.numel() > 0 else xyz_canonical.device)
                        xyz_min = 0.0
                        xyz_max = 0.0
                    # Check numerical stability in inference diagnostics.
                    has_nan = (torch.isnan(xyz).any() if xyz.numel() > 0 else False) or (torch.isnan(scaling).any() if scaling.numel() > 0 else False) or (torch.isnan(rot).any() if rot.numel() > 0 else False)
                    has_inf = (torch.isinf(xyz).any() if xyz.numel() > 0 else False) or (torch.isinf(scaling).any() if scaling.numel() > 0 else False) or (torch.isinf(rot).any() if rot.numel() > 0 else False)
                    scaling_min = scaling.min().item() if scaling.numel() > 0 else 0.0
                    scaling_max = scaling.max().item() if scaling.numel() > 0 else 0.0
                    print(f"[INFER] Final xyz diff from canonical: mean={xyz_diff.item():.6f}, max={xyz_diff_max.item():.6f}")
                    print(f"[INFER] xyz.shape={xyz.shape}, scaling.shape={scaling.shape}, rot.shape={rot.shape}")
                    print(f"[INFER] has_nan={has_nan}, has_inf={has_inf}")
                    print(f"[INFER] xyz range: min={xyz_min:.3f}, max={xyz_max:.3f}")
                    print(f"[INFER] scaling range: min={scaling_min:.6f}, max={scaling_max:.6f}")

            with torch.no_grad():

                if d_xyz.numel() > 0:
                    deform_norm = torch.norm(d_xyz, dim=-1)
                    deform_norm_mean = float(deform_norm.mean().item())
                    deform_norm_max = float(deform_norm.max().item())
                    deform_norm_std = float(deform_norm.std().item())
                else:
                    deform_norm_mean = 0.0
                    deform_norm_max = 0.0
                    deform_norm_std = 0.0
                if rot.numel() > 0:
                    rot_norm = torch.norm(rot, dim=-1)
                    rot_norm_mean = float(rot_norm.mean().item())
                    rot_norm_min = float(rot_norm.min().item())
                    rot_norm_max = float(rot_norm.max().item())
                else:
                    rot_norm_mean = 0.0
                    rot_norm_min = 0.0
                    rot_norm_max = 0.0
                pc.deform_stats = {
                    "deform_norm_mean": deform_norm_mean,
                    "deform_norm_max": deform_norm_max,
                    "deform_norm_std": deform_norm_std,
                    "rot_norm_mean": rot_norm_mean,
                    "rot_norm_min": rot_norm_min,
                    "rot_norm_max": rot_norm_max
                }
                if hasattr(pc.deform, 'debug_stats'):
                    pc.deform_stats.update(pc.deform.debug_stats)
            if hasattr(pc.deform, "last_reg") and pc.deform.last_reg is not None:
                pc.last_deform_reg = pc.deform.last_reg

                if debug_enabled and not hasattr(pc, '_debug_deform_reg_set'):
                    pc._debug_deform_reg_set = True
                    print(f"[GenerateNeuralGaussians] Setting pc.last_deform_reg = {pc.last_deform_reg.item():.8f}")
            else:
                pc.last_deform_reg = None
                if debug_enabled and not hasattr(pc, '_debug_deform_reg_none'):
                    pc._debug_deform_reg_none = True
                    print(f"[GenerateNeuralGaussians] WARNING: pc.deform.last_reg is None! Deformation regularization will not be applied!")

            # Clear intermediate variables after statistics and regularization are recorded.
            del xyz_input, d_xyz, d_rotation, d_scaling, time_input
        else:
            # Non-rigid deformation is disabled; apply rigid transform only.
            if debug_enabled and is_infer and not hasattr(pc, '_debug_infer_no_deform_printed'):
                pc._debug_infer_no_deform_printed = True
                print(f"[INFER] WARNING: use_deform=False! stage={stage}, xyz_canonical.shape[0]={xyz_canonical.shape[0]}, enable_non_rigid={enable_non_rigid}")
                print(f"[INFER] Conditions: stage=='fine'={stage=='fine'}, xyz_canonical.shape[0]>100={xyz_canonical.shape[0]>100}, enable_non_rigid={enable_non_rigid}")
                print(f"[INFER] This means non-rigid deformation is NOT being applied! But rigid transform will still be applied.")

            obj_rot_masked = obj_rot[visible_mask]
            obj_trans_masked = obj_trans_input[visible_mask]

            obj_rot_selected = obj_rot_masked[anchor_indices_repeat]  # (num_gaussians, 3, 3)
            obj_trans_selected = obj_trans_masked[anchor_indices_repeat]  # (num_gaussians, 3)

            xyz = torch.einsum('bij, bj -> bi', obj_rot_selected, xyz_canonical) + obj_trans_selected

        # Recompute normals after deformation or rigid transform.
        rotations_mat = quaternion_to_matrix(rot)
        min_scales = torch.argmin(scaling, dim=-1)
        indices = torch.arange(min_scales.shape[0], device=device)
        normals = rotations_mat[indices, :, min_scales]


        dir_pp = xyz - camera_center.expand(xyz.shape[0], -1)
        dir_pp_norm = torch.norm(dir_pp, dim=1, keepdim=True)
        dir_pp_normalized = dir_pp / (dir_pp_norm + 1e-8)
        dotprod = torch.sum(-dir_pp_normalized * normals, dim=1, keepdim=True)
        normals = torch.where(dotprod >= 0, normals, -normals)


        shs_features = shs

        # Write per-Gaussian opacity values back into the full per-offset tensor.
        if use_deform and enable_non_rigid:
            if opacity.dim() == 1:
                opacity = opacity.unsqueeze(-1)
            elif opacity.dim() > 2:
                opacity = opacity.squeeze(-1).unsqueeze(-1)

            num_gaussians = xyz.shape[0]

            if opacity.shape[0] != num_gaussians:
                raise ValueError(f"opacity.shape[0]={opacity.shape[0]} != num_gaussians={num_gaussians}, xyz.shape[0]={xyz.shape[0]}")

            neural_opacity_clone = neural_opacity.clone()


            opacity_thresh = float(getattr(hyper, "opacity_thresh", 0.01)) if hyper is not None else 0.01
            if 'original_mask_before_deform' in locals() and original_mask_before_deform is not None:
                original_valid_mask = original_mask_before_deform.view(-1)
            else:
                original_valid_mask = (neural_opacity > opacity_thresh).view(-1)

            idx = torch.nonzero(original_valid_mask, as_tuple=False).squeeze(1)
            n_assign = min(idx.shape[0], opacity.shape[0])
            if n_assign > 0:
                neural_opacity_clone[idx[:n_assign]] = opacity[:n_assign]

            mask = original_valid_mask
        else:
            neural_opacity_clone = neural_opacity.clone()
            opacity_thresh = float(getattr(hyper, "opacity_thresh", 0.01)) if hyper is not None else 0.01
            valid_mask = mask_init.view(-1) if 'mask_init' in locals() else (neural_opacity > opacity_thresh).view(-1)
            if opacity.dim() == 1:
                opacity = opacity.unsqueeze(-1)
            elif opacity.dim() > 2:
                opacity = opacity.squeeze(-1).unsqueeze(-1)
            idx = torch.nonzero(valid_mask, as_tuple=False).squeeze(1)
            n_assign = min(idx.shape[0], opacity.shape[0])
            if n_assign > 0:
                neural_opacity_clone[idx[:n_assign]] = opacity[:n_assign]
            mask = valid_mask

        with torch.no_grad():
            xyz_shape0 = int(xyz.shape[0])
            color_shape0 = int(color.shape[0])
            opacity_shape0 = int(opacity.shape[0])
            scaling_shape0 = int(scaling.shape[0])
            rot_shape0 = int(rot.shape[0])
            mask_sum = int(mask.sum().item()) if mask.numel() > 0 else 0
            neural_opacity_shape0 = int(neural_opacity.shape[0]) if neural_opacity.numel() > 0 else 0
            shapes_match = (xyz_shape0 == color_shape0 == opacity_shape0 == scaling_shape0 == rot_shape0)



        if not shapes_match:
            raise ValueError(f"Shape mismatch in return values: xyz.shape[0]={xyz_shape0}, color.shape[0]={color_shape0}, opacity.shape[0]={opacity_shape0}, scaling.shape[0]={scaling_shape0}, rot.shape[0]={rot_shape0}, mask.sum()={mask_sum}. This will cause gradient shape mismatch!")

        if is_training:
            return xyz, color, opacity, scaling, rot, neural_opacity, mask, normals, semantics, shs_features
        else:
            return xyz, color, opacity, scaling, rot, normals, semantics, shs_features

    def color_to_sh_features(self, color, max_sh_degree=0, device=None):
        """
        Convert RGB colors to SH feature tensors with only the DC component set.

        Args:
            color: RGB tensor with shape [N, 3].
            max_sh_degree: maximum spherical harmonic degree.
            device: output device.

        Returns:
            SH tensor with shape [N, (max_sh_degree + 1) ** 2, 3].
        """
        if device is None and isinstance(color, torch.Tensor):
            device = color.device

        batch_size = color.shape[0]
        shs_feature = torch.zeros(batch_size, 3, (max_sh_degree+1)**2, dtype=torch.float, device=device)

        # RGB2SH for the DC component.
        sh_color = (color - 0.5) / 0.28209479177387814
        shs_feature[:, :, 0] = sh_color


        shs_feature[:, :, 1:] = 0.0


        # shs_features shape: (N, (max_sh_degree+1)^2, 3)
        shs_features = shs_feature.transpose(1, 2).contiguous()

        return shs_features

    def sh_features_to_color(self, shs_features, max_sh_degree=0):
        """
        Convert SH features back to RGB using the DC component.

        Args:
            shs_features: SH tensor.
            max_sh_degree: maximum spherical harmonic degree.

        Returns:
            RGB tensor with shape [N, 3].
        """

        if shs_features.shape[1] == (max_sh_degree+1)**2:
            features_dc = shs_features[:, 0:1, :]
        else:
            features_dc = shs_features[:, 0:1, :]


        sh_color = features_dc.transpose(1, 2)

        C0 = 0.28209479177387814
        color = sh_color[:, :, 0] * C0 + 0.5

        color = torch.clamp(color, 0.0, 1.0)

        return color

def rotate_xyz(points, angle_x=0, angle_y=0, angle_z=0):
    ax = torch.deg2rad(torch.tensor(angle_x, device=points.device, dtype=points.dtype))
    ay = torch.deg2rad(torch.tensor(angle_y, device=points.device, dtype=points.dtype))
    az = torch.deg2rad(torch.tensor(angle_z, device=points.device, dtype=points.dtype))
    cx, sx = torch.cos(ax), torch.sin(ax)
    cy, sy = torch.cos(ay), torch.sin(ay)
    cz, sz = torch.cos(az), torch.sin(az)
    Rx = torch.stack([torch.stack([torch.ones_like(cx), torch.zeros_like(cx), torch.zeros_like(cx)]),
                     torch.stack([torch.zeros_like(cx), cx, -sx]),
                     torch.stack([torch.zeros_like(cx), sx, cx])], dim=-1)
    Ry = torch.stack([torch.stack([cy, torch.zeros_like(cy), sy]),
                     torch.stack([torch.zeros_like(cy), torch.ones_like(cy), torch.zeros_like(cy)]),
                     torch.stack([-sy, torch.zeros_like(cy), cy])], dim=-1)
    Rz = torch.stack([torch.stack([cz, -sz, torch.zeros_like(cz)]),
                     torch.stack([sz, cz, torch.zeros_like(cz)]),
                     torch.stack([torch.zeros_like(cz), torch.zeros_like(cz), torch.ones_like(cz)]),], dim=-1)
    R = Rz @ Ry @ Rx
    rotated_points = points @ R.T
    return rotated_points
