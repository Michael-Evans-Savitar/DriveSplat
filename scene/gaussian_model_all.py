#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
from datetime import timedelta
import time
import torch
from functools import reduce
import numpy as np
from torch_scatter import scatter_max
from utils.general_utils import inverse_sigmoid, get_expon_lr_func
from torch import nn
import os
from utils.system_utils import mkdir_p
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
from utils.graphics_utils import BasicPointCloud
from utils.general_utils import strip_symmetric, build_scaling_rotation
from scene.embedding import Embedding
from einops import repeat
import math
from scene.gaussian_model_actor import GaussianModelActor
from scene.gaussian_model import GaussianModel
from scene.gaussian_model_bkgd import GaussianModelBkgd
from scene.hash_encoding import HashEncoding
from utils.general_utils import quaternion_to_matrix, build_scaling_rotation, strip_symmetric, quaternion_raw_multiply, startswith_any, matrix_to_quaternion
from bidict import bidict
from scene.actor_pose import ActorPose
from scene.camera_pose import PoseCorrection
from scene.color_correction import ColorCorrection
from typing import Optional


class SceneGraphNode:
    """
    ,(/).
    ,.
    """
    def __init__(self, name, node_type="object", parent=None):
        self.name = name
        self.node_type = node_type  # "background" / "object" / "root"
        self.parent = parent
        self.children = []
        self.center = None
        self.extent = None
        self.visible = True
        self.dirty = True

    def update(self, center=None, extent=None, visible=None):
        if center is not None:
            self.center = center
        if extent is not None:
            self.extent = extent
        if visible is not None:
            self.visible = bool(visible)
        self.dirty = False

class DriveSplatModel(nn.Module):

    def setup_functions(self, sh_degree, args, feat_dim, n_offsets, fork, use_feat_bank, appearance_dim, add_opacity_dist, add_cov_dist, add_color_dist, add_level, visible_threshold, dist2level, base_layer, progressive, extend):
        def build_covariance_from_scaling_rotation(scaling, scaling_modifier, rotation):
            L = build_scaling_rotation(scaling_modifier * scaling, rotation)
            actual_covariance = L @ L.transpose(1, 2)
            symm = strip_symmetric(actual_covariance)
            return symm

        self.scaling_activation = torch.exp
        self.scaling_inverse_activation = torch.log

        self.covariance_activation = build_covariance_from_scaling_rotation

        self.opacity_activation = torch.sigmoid
        self.inverse_opacity_activation = inverse_sigmoid

        self.rotation_activation = torch.nn.functional.normalize
        if args.include_obj:
            obj_tracklets = self.metadata["obj_tracklets"]
            obj_info = self.metadata["obj_meta"]
            tracklet_timestamps = self.metadata["tracklet_timestamps"]
            camera_timestamps = self.metadata["camera_timestamps"]
        else:
            obj_tracklets = None
            obj_info = None
            tracklet_timestamps = None
            camera_timestamps = None
        self.model_name_id = bidict()
        self.obj_list = []
        self.models_num = 0
        self.obj_info = obj_info
        # Build background model
        if args.include_bkgd:
            self.background = GaussianModelBkgd(model_name='background', sh_degree=sh_degree, args=args, feat_dim=feat_dim, n_offsets=n_offsets, fork=fork, use_feat_bank=use_feat_bank, appearance_dim=appearance_dim, add_opacity_dist=add_opacity_dist, add_cov_dist=add_cov_dist, add_color_dist=add_color_dist, add_level=add_level, visible_threshold=visible_threshold, dist2level=dist2level, base_layer=base_layer, progressive=progressive, extend=extend)
            self.model_name_id['background'] = 0
            self.models_num += 1
        # Build object model
        if args.include_obj:
            for track_id, obj_meta in self.obj_info.items():
                model_name = f'obj_{track_id:03d}'
                setattr(self, model_name, GaussianModelActor(model_name=model_name, obj_meta=obj_meta, args=args, sh_degree=sh_degree, feat_dim=feat_dim, n_offsets=n_offsets, fork=fork, use_feat_bank=use_feat_bank, appearance_dim=appearance_dim, add_opacity_dist=add_opacity_dist, add_cov_dist=add_cov_dist, add_color_dist=add_color_dist, add_level=add_level, visible_threshold=visible_threshold, dist2level=dist2level, base_layer=base_layer, progressive=progressive, extend=extend))
                self.model_name_id[model_name] = self.models_num
                self.obj_list.append(model_name)
                self.models_num += 1
        # Build actor model
        if args.include_obj:
            self.actor_pose = ActorPose(args, obj_tracklets, tracklet_timestamps, camera_timestamps, obj_info)
        else:
            self.actor_pose = None

        # Build pose correction
        if self.use_pose_correction:
            self.pose_correction = PoseCorrection(self.metadata, args)
        else:
            self.pose_correction = None

        # Build color correction
        if self.use_color_correction:
            self.color_correction = ColorCorrection(self.metadata, args)
        else:
            self.color_correction = None


    def __init__(self, metadata, sh_degree, args,
                 feat_dim: int=32,
                 n_offsets: int=5,
                 fork: int=2,
                 use_feat_bank : bool = False,
                 appearance_dim : int = 32,
                 add_opacity_dist : bool = False,
                 add_cov_dist : bool = False,
                 add_color_dist : bool = False,
                 add_level: bool = False,
                 visible_threshold: float = -1,
                 dist2level: str = 'round',
                 base_layer: int = 10,
                 progressive: bool = True,
                 extend: float = 1.1
                 ):
        # semantic
        super().__init__()
        self.metadata = metadata
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0
        num_classes = 1 if args.use_semantic else 0
        self.num_classes = num_classes
        self.semantic_mode = args.semantic_mode
        assert self.semantic_mode in ['logits', 'probabilities']

        self.feat_dim = feat_dim
        self.view_dim = 3
        self.n_offsets = n_offsets
        self.fork = fork
        self.use_feat_bank = use_feat_bank
        self._normal = torch.empty(0)
        self._features_dc = torch.empty(0)
        self._features_rest = torch.empty(0)
        self.appearance_dim = appearance_dim
        self.embedding_appearance = None
        self.add_opacity_dist = add_opacity_dist
        self.use_pose_correction = args.use_pose_correction
        self.use_color_correction = args.use_color_correction
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist
        self.add_level = add_level
        self.progressive = progressive

        self.sub_pos_offsets = torch.tensor([[i % fork, (i // fork) % fork, i // (fork * fork)] for i in range(fork**3)]).float().cuda()
        self.extend = extend
        self.visible_threshold = visible_threshold
        self.dist2level = dist2level
        self.base_layer = base_layer

        self.start_step = 0
        self.end_step = 0

        self._anchor = torch.empty(0)
        self._level = torch.empty(0)
        self._offset = torch.empty(0)
        self._anchor_feat = torch.empty(0)
        self.opacity_accum = torch.empty(0)
        self._scaling = torch.empty(0)
        self._rotation = torch.empty(0)
        self._opacity = torch.empty(0)
        self._semantic = torch.empty(0)
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)

        self.anchor_demon = torch.empty(0)

        self.optimizer = None
        self.percent_dense = 0
        self.spatial_lr_scale = 0
        self.setup_functions(sh_degree, args, feat_dim, n_offsets, fork, use_feat_bank, appearance_dim, add_opacity_dist, add_cov_dist, add_color_dist, add_level, visible_threshold, dist2level, base_layer, progressive, extend)

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.level_dim = 1 if self.add_level else 0

        self.include_obj = args.include_obj
        self.include_background = args.include_bkgd
        self.include_sky = args.include_sky
        self.flip_prob = args.flip_prob
        self.flip_axis = 1
        self.flip_matrix = torch.eye(3).float().cuda() * -1
        self.flip_matrix[self.flip_axis, self.flip_axis] = 1
        self.flip_matrix = matrix_to_quaternion(self.flip_matrix.unsqueeze(0))

        if self.use_feat_bank:
            self.mlp_feature_bank = nn.Sequential(
                    nn.Linear(self.view_dim+self.level_dim, self.feat_dim),
                    nn.ReLU(True),
                    nn.Linear(self.feat_dim, 3),
                    nn.Softmax(dim=1)
                ).cuda()

        self.mlp_opacity = nn.Sequential(
                nn.Linear(self.feat_dim+self.view_dim+self.opacity_dist_dim+self.level_dim, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, self.n_offsets),
                nn.Tanh()
            ).cuda()

        self.mlp_cov = nn.Sequential(
                nn.Linear(self.feat_dim+self.view_dim+self.cov_dist_dim+self.level_dim, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, 7*self.n_offsets),
            ).cuda()

        self.mlp_color = nn.Sequential(
                nn.Linear(self.feat_dim+self.view_dim+self.color_dist_dim+self.level_dim+self.appearance_dim, self.feat_dim),
                nn.ReLU(True),
                nn.Linear(self.feat_dim, 3*self.n_offsets),
                nn.Sigmoid()
            ).cuda()


        self._init_scene_graph()

        #  include_list
        self.include_list = list(self.model_name_id.keys())

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()
        #  eval
        for model_name in self.model_name_id.keys():
            model = getattr(self, model_name)
            model.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.use_feat_bank:
            self.mlp_feature_bank.train()
        #  train
        for model_name in self.model_name_id.keys():
            model = getattr(self, model_name)
            model.train()

    def save_state_dict(self, is_final):

        state_dict = {}
        for model_name in self.model_name_id.keys():
            model: GaussianModel = getattr(self, model_name)
            state_dict[model_name] = model.state_dict(is_final)
        if self.actor_pose is not None:
            state_dict['actor_pose'] = self.actor_pose.save_state_dict(is_final)
        if self.color_correction is not None:
            state_dict['color_correction'] = self.color_correction.save_state_dict(is_final)
        if self.pose_correction is not None:
            state_dict['pose_correction'] = self.pose_correction.save_state_dict(is_final)

        return state_dict

    def capture(self, exclude_list=[]):
        """
        Checkpoint format for training resume.
        Returns a dict {model_name: model.capture(), ...} and optional extra modules.
        """
        state = {}
        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            if hasattr(model, "capture"):
                state[model_name] = model.capture()
        if self.actor_pose is not None:
            # keep optimizer out to reduce incompatibility; actor_pose is lightweight to re-opt if needed
            state["actor_pose"] = self.actor_pose.save_state_dict(is_final=False)
        if self.color_correction is not None:
            state["color_correction"] = self.color_correction.save_state_dict(is_final=False)
        if self.pose_correction is not None:
            state["pose_correction"] = self.pose_correction.save_state_dict(is_final=False)
        return state

    def restore(self, model_args, training_args):
        # Newer checkpoints: dict per sub-model
        if isinstance(model_args, dict):
            for model_name in self.model_name_id.keys():
                if model_name not in model_args:
                    continue
                model: GaussianModel = getattr(self, model_name)
                model.restore(model_args[model_name], training_args)
            if self.actor_pose is not None and "actor_pose" in model_args:
                try:
                    self.actor_pose.load_state_dict(model_args["actor_pose"])
                except Exception:
                    pass
            if self.color_correction is not None and "color_correction" in model_args:
                try:
                    self.color_correction.load_state_dict(model_args["color_correction"])
                except Exception:
                    pass
            if self.pose_correction is not None and "pose_correction" in model_args:
                try:
                    self.pose_correction.load_state_dict(model_args["pose_correction"])
                except Exception:
                    pass
            return

        # Legacy checkpoints: a single tuple shared by one model (kept for backward compatibility)
        for model_name in self.model_name_id.keys():
            model: GaussianModel = getattr(self, model_name)
            model.restore(model_args, training_args)

    def load_state_dict(self, state_dict):
        for model_name in self.model_name_id.keys():
            model: GaussianModel = getattr(self, model_name)
            model.load_state_dict(state_dict[model_name])

        if self.actor_pose is not None:
            self.actor_pose.load_state_dict(state_dict['actor_pose'])

    def save_state_dict(self, is_final, exclude_list=[]):
        state_dict = dict()

        for model_name in self.model_name_id.keys():
            if startswith_any(model_name, exclude_list):
                continue
            model: GaussianModel = getattr(self, model_name)
            state_dict[model_name] = model.state_dict(is_final)

        if self.actor_pose is not None:
            state_dict['actor_pose'] = self.actor_pose.save_state_dict(is_final)

        return state_dict

    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        scalings = []

        if self.get_visibility('background'):
            scaling_bkgd = self.background.get_scaling
            scalings.append(scaling_bkgd)

        for obj_name in self.graph_obj_list:
            obj_model: GaussianModelActor = getattr(self, obj_name)

            scaling = obj_model.get_scaling

            scalings.append(scaling)

        scalings = torch.cat(scalings, dim=0)
        self._scaling = scalings
        return scalings

    @property
    def get_rotation(self):
        rotations = []
        is_training = self.get_color_mlp.training
        if self.get_visibility('background'):
            rotations_bkgd = self.background.get_rotation
            if self.use_pose_correction:
                rotations_bkgd = self.pose_correction.correct_gaussian_rotation(self.viewpoint_camera, rotations_bkgd, is_training)
            rotations.append(rotations_bkgd)

        if len(self.graph_obj_list) > 0 and len(self.obj_rots) > 0:
            rotations_local = []
            for i, obj_name in enumerate(self.graph_obj_list):
                obj_model: GaussianModelActor = getattr(self, obj_name)
                rotation_local = obj_model.get_rotation
                rotations_local.append(rotation_local)

            if len(rotations_local) > 0:
                rotations_local = torch.cat(rotations_local, dim=0)
                rotations_local = rotations_local.clone()
                if hasattr(self, 'flip_mask') and len(self.flip_mask) > 0 and len(self.flip_mask) == len(rotations_local):
                    rotations_local[self.flip_mask] = quaternion_raw_multiply(self.flip_matrix, rotations_local[self.flip_mask])
                if len(self.obj_rots) == len(rotations_local):
                    rotations_obj = quaternion_raw_multiply(self.obj_rots, rotations_local)
                    rotations_obj = torch.nn.functional.normalize(rotations_obj)
                    rotations.append(rotations_obj)

        rotations = torch.cat(rotations, dim=0)
        self._rotation = rotations
        return rotations

    @property
    def get_anchor(self):
        anchors = []
        if self.get_visibility('background'):
            is_training = self.get_color_mlp.training
            anchor_bkgd = self.background.get_anchor
            if self.use_pose_correction:
                anchor_bkgd = self.pose_correction.correct_gaussian_xyz(self.viewpoint_camera, anchor_bkgd, is_training)
            anchors.append(anchor_bkgd)

        if len(self.graph_obj_list) > 0 and len(self.obj_rots) > 0 and len(self.obj_trans) > 0:
            anchors_local = []

            for i, obj_name in enumerate(self.graph_obj_list):
                obj_model: GaussianModelActor = getattr(self, obj_name)
                anchor_local = obj_model.get_anchor
                anchors_local.append(anchor_local)

            if len(anchors_local) > 0:
                anchors_local = torch.cat(anchors_local, dim=0)
                anchors_local = anchors_local.clone()
                if hasattr(self, 'flip_mask') and len(self.flip_mask) > 0 and len(self.flip_mask) == len(anchors_local):
                    anchors_local[self.flip_mask, self.flip_axis] *= -1
                if len(self.obj_rots) == len(anchors_local) and len(self.obj_trans) == len(anchors_local):
                    obj_rots = quaternion_to_matrix(self.obj_rots)
                    anchors_obj = torch.einsum('bij, bj -> bi', obj_rots, anchors_local) + self.obj_trans
                    anchors.append(anchors_obj)

        if len(anchors) > 0:
            anchors = torch.cat(anchors, dim=0)
            self._anchor = anchors
        else:
            self._anchor = torch.empty(0)
        return anchors

    @property
    def get_level(self):
        levels = []
        if self.get_visibility('background'):
            level_bkgd = self.background.get_level
            levels.append(level_bkgd)

        for obj_name in self.graph_obj_list:
            obj_model: GaussianModelActor = getattr(self, obj_name)

            level = obj_model.get_level

            levels.append(level)

        levels = torch.cat(levels, dim=0)
        self._level = levels
        return levels

    @property
    def get_extra_level(self):
        extra_levels = []
        if self.get_visibility('background'):
            extra_level_bkgd = self.background.get_extra_level
            extra_levels.append(extra_level_bkgd)

        for obj_name in self.graph_obj_list:
            obj_model: GaussianModelActor = getattr(self, obj_name)

            extra_level = obj_model.get_extra_level

            extra_levels.append(extra_level)

        extra_levels = torch.cat(extra_levels, dim=0)
        self._extra_level = extra_levels
        return self._extra_level

    @property
    def get_features(self):
        features = []

        if self.get_visibility('background'):
            features_bkgd = self.background.get_features
            features.append(features_bkgd)

        for i, obj_name in enumerate(self.graph_obj_list):
            obj_model: GaussianModelActor = getattr(self, obj_name)
            feature_obj = obj_model.get_features_fourier(self.frame)
            features.append(feature_obj)

        features = torch.cat(features, dim=0)

        return features

    @property
    def get_opacity(self):
        opacities = []

        if self.get_visibility('background'):
            opacity_bkgd = self.background.get_opacity
            opacities.append(opacity_bkgd)

        for obj_name in self.graph_obj_list:
            obj_model: GaussianModelActor = getattr(self, obj_name)

            opacity = obj_model.get_opacity

            opacities.append(opacity)

        opacities = torch.cat(opacities, dim=0)
        self._opacity = opacities
        return opacities

    @property
    def get_anchor_feat(self):
        anchor_feats = []

        if self.get_visibility('background'):
            anchor_feat_bkgd = self.background.get_anchor_feat
            anchor_feats.append(anchor_feat_bkgd)

        for obj_name in self.graph_obj_list:
            obj_model: GaussianModelActor = getattr(self, obj_name)

            anchor_feat = obj_model.get_anchor_feat

            anchor_feats.append(anchor_feat)

        anchor_feats = torch.cat(anchor_feats, dim=0)
        self._anchor_feat = anchor_feats
        return anchor_feats

    @property
    def get_opacity_mlp(self):
        return self.mlp_opacity

    @property
    def get_cov_mlp(self):
        return self.mlp_cov

    @property
    def get_color_mlp(self):
        return self.mlp_color

    @property
    def get_featurebank_mlp(self):
        mlp_feature_banks = []
        if self.get_visibility('background'):
            mlp_feature_bank_bkgd = self.background.get_featurebank_mlp
            mlp_feature_banks.append(mlp_feature_bank_bkgd)
        for obj_name in self.graph_obj_list:
            obj_model: GaussianModelActor = getattr(self, obj_name)
            mlp_feature_bank = obj_model.get_featurebank_mlp
            mlp_feature_banks.append(mlp_feature_bank)
        mlp_feature_banks = torch.cat(mlp_feature_banks, dim=0)
        self.mlp_feature_bank = mlp_feature_banks
        return mlp_feature_banks

    def set_appearance(self, num_cameras):
        if self.get_visibility('background'):
            self.background.set_appearance(num_cameras)

        for obj_name in self.obj_list:
            obj_model: GaussianModelActor = getattr(self, obj_name)
            obj_model.set_appearance(num_cameras)

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def set_coarse_interval(self, coarse_iter, coarse_factor):
        for model_name in self.model_name_id.keys():
            model: GaussianModel = getattr(self, model_name)
            if model_name in ['background', 'sky']:
                model.set_coarse_interval(coarse_iter, coarse_factor)
            else:
                model.set_coarse_interval(coarse_iter, coarse_factor)

    def set_level(self, points, cameras, scales, dist_ratio=0.95, init_level=-1, levels=-1, model_path=None):
        for model_name in self.model_name_id.keys():
            model: GaussianModel = getattr(self, model_name)
            if model_name in ['background', 'sky']:
                model.set_level(points, cameras, scales, dist_ratio, init_level, levels=levels)
            elif model.obj_class == 'pedestrian':
                model.set_level(points, cameras, scales, dist_ratio, init_level, levels=levels, model_path=model_path)
            else:
                model.set_level(points, cameras, scales, dist_ratio, init_level, levels=levels, model_path=model_path)

    def lod_tree_sample(self, data, init_pos):
        torch.cuda.synchronize(); t0 = time.time()
        self.positions = torch.empty(0, 3).float().cuda()
        self._level = torch.empty(0).int().cuda()
        for cur_level in range(self.levels):
            cur_size = self.voxel_size/(float(self.fork) ** cur_level)
            new_positions = torch.unique(torch.round((data - init_pos) / cur_size), dim=0) * cur_size + init_pos
            new_level = torch.ones(new_positions.shape[0], dtype=torch.int, device="cuda") * cur_level
            self.positions = torch.concat((self.positions, new_positions), dim=0)
            self._level = torch.concat((self._level, new_level), dim=0)
        torch.cuda.synchronize(); t1 = time.time()
        time_diff = t1 - t0
        print(f"Building LOD tree time: {int(time_diff // 60)} min {time_diff % 60} sec")

    def create_from_pcd(self, points, spatial_lr_scale, first_cam_center=None, logger=None, model_path=None):
        for model_name in self.model_name_id.keys():
            model: GaussianModel = getattr(self, model_name)
            if model_name in ['background', 'sky']:
                model.create_from_pcd(points, spatial_lr_scale, first_cam_center, logger)
            else:
                model.create_from_pcd(spatial_lr_scale, logger, model_path=model_path)


    def map_to_int_level(self, pred_level, cur_level):
        if self.dist2level=='floor':
            int_level = torch.floor(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='round':
            int_level = torch.round(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='ceil':
            int_level = torch.ceil(pred_level).int()
            int_level = torch.clamp(int_level, min=0, max=cur_level)
        elif self.dist2level=='progressive':
            pred_level = torch.clamp(pred_level+1.0, min=0.9999, max=cur_level + 0.9999)
            int_level = torch.floor(pred_level).int()
            self._prog_ratio = torch.frac(pred_level).unsqueeze(dim=1)
            self.transition_mask = (self._level.squeeze(dim=1) == int_level)
        else:
            raise ValueError(f"Unknown dist2level: {self.dist2level}")

        return int_level

    def weed_out(self, anchor_positions, anchor_levels):
        visible_count = torch.zeros(anchor_positions.shape[0], dtype=torch.int, device="cuda")
        for cam in self.cam_infos:
            cam_center, scale = cam[:3], cam[3]
            dist = torch.sqrt(torch.sum((anchor_positions - cam_center)**2, dim=1)) * scale
            pred_level = torch.log2(self.standard_dist/dist)/math.log2(self.fork)
            int_level = self.map_to_int_level(pred_level, self.levels - 1)
            visible_count += (anchor_levels <= int_level).int()
        visible_count = visible_count/len(self.cam_infos)
        weed_mask = (visible_count > self.visible_threshold)
        mean_visible = torch.mean(visible_count)
        return anchor_positions[weed_mask], anchor_levels[weed_mask], mean_visible, weed_mask

    def set_anchor_mask(self, cam_center, iteration, resolution_scale):

        get_anchor_mask = []
        if self.get_visibility('background'):
            model: GaussianModel = self.background
            model.set_anchor_mask(cam_center, iteration, resolution_scale)
            get_anchor_mask.append(model._anchor_mask)
        if len(self.graph_obj_list) > 0:
            anchor_mask_local = []
            for i, obj_name in enumerate(self.graph_obj_list):
                obj_model: GaussianModelActor = getattr(self, obj_name)
                obj_model.set_anchor_mask(cam_center, iteration, resolution_scale)
                anchor_mask_local.append(obj_model._anchor_mask)
            anchor_mask_local = torch.cat(anchor_mask_local, dim=0)
            get_anchor_mask.append(anchor_mask_local)
        get_anchor_mask = torch.cat(get_anchor_mask, dim=0)
        self._anchor_mask = get_anchor_mask

    def set_obj_anchor_mask(self, cam_center, iteration, resolution_scale):

        get_anchor_mask = []
        if len(self.graph_obj_list) > 0:
            anchor_mask_local = []
            for i, obj_name in enumerate(self.graph_obj_list):
                obj_model: GaussianModelActor = getattr(self, obj_name)
                obj_model.set_anchor_mask(cam_center, iteration, resolution_scale)
                anchor_mask_local.append(obj_model._anchor_mask)
            anchor_mask_local = torch.cat(anchor_mask_local, dim=0)
            get_anchor_mask.append(anchor_mask_local)
            get_anchor_mask = torch.cat(get_anchor_mask, dim=0)
            self.obj_anchor_mask = get_anchor_mask

    def set_anchor_mask_perlevel(self, cam_center, resolution_scale, cur_level):
        for model_name in self.model_name_id.keys():
            model: GaussianModel = getattr(self, model_name)
            model.set_anchor_mask_perlevel(cam_center, resolution_scale, cur_level)

    def set_obj_anchor_mask_perlevel(self, cam_center, resolution_scale, cur_level):
        for model_name in self.model_name_id.keys():
            if model_name != 'background':
                model: GaussianModel = getattr(self, model_name)
                model.set_anchor_mask_perlevel(cam_center, resolution_scale, cur_level)


    def training_setup(self, training_args):
        self.active_sh_degree = 0

        for model_name in self.model_name_id.keys():
            model: GaussianModel = getattr(self, model_name)
            model.training_setup(training_args)

        if self.color_correction is not None:
            self.color_correction.training_setup(training_args)

        if self.pose_correction is not None:
            self.pose_correction.training_setup(training_args)

        if self.actor_pose is not None:
            self.actor_pose.training_setup(training_args)


    def update_learning_rate(self, iteration):
        for model_name in self.model_name_id.keys():
            model: GaussianModel = getattr(self, model_name)
            model.update_learning_rate(iteration)
        if self.actor_pose is not None:
            self.actor_pose.update_learning_rate(iteration)
        if self.color_correction is not None:
            self.color_correction.update_learning_rate(iteration)
        if self.pose_correction is not None:
            self.pose_correction.update_learning_rate(iteration)

    def construct_list_of_attributes(self):
        l = []
        l.append('x')
        l.append('y')
        l.append('z')
        l.append('level')
        l.append('extra_level')
        l.append('info_0')
        l.append('info_1')
        for i in range(self._offset.shape[1]*self._offset.shape[2]):
            l.append('f_offset_{}'.format(i))
        for i in range(self._anchor_feat.shape[1]):
            l.append('f_anchor_feat_{}'.format(i))
        l.append('opacity')
        for i in range(self._scaling.shape[1]):
            l.append('scale_{}'.format(i))
        for i in range(self._rotation.shape[1]):
            l.append('rot_{}'.format(i))
        for i in range(self._semantic.shape[1]):
            l.append('semantic_{}'.format(i))
        return l

    def save_deform_weights(self, path, iteration):
        for model_name in self.model_name_id.keys():
            model: GaussianModel = getattr(self, model_name)
            if model_name != 'background' and model_name != 'sky':
                if model.obj_class == "pedestrian":
                    model.save_deform_weights(path, iteration)

    def load_deform_weights(self, path, iteration):
        # (vehicle)deform
        for model_name in self.model_name_id.keys():
            model: GaussianModel = getattr(self, model_name)
            if model_name != 'background' and model_name != 'sky':
                if hasattr(model, 'deformable') and model.deformable:
                    model.load_deform_weights(path, iteration)
                elif hasattr(model, 'obj_class') and model.obj_class == 'pedestrian':
                    # ,pedestrian
                    model.load_deform_weights(path, iteration)

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        print(f"Saving ply to {path}")


        all_xyz = []
        all_levels = []
        all_extra_levels = []
        all_anchor_feats = []
        all_offsets = []
        all_opacities = []
        all_scales = []
        all_rots = []
        all_semantics = []
        all_model_ids = []
        all_main_directions = []
        all_init_pos = []
        all_bounds = []


        for model_name in self.model_name_id.keys():
            model = getattr(self, model_name)
            model_id = self.model_name_id[model_name]
            num_points = model._anchor.shape[0]
            print(f"model_name: {model_name}, points: {num_points}")

            all_xyz.append(model._anchor.detach().cpu().numpy())
            all_levels.append(model._level.detach().cpu().numpy())
            all_extra_levels.append(model._extra_level.unsqueeze(dim=1).detach().cpu().numpy())
            all_anchor_feats.append(model._anchor_feat.detach().cpu().numpy())
            all_offsets.append(model._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy())
            all_opacities.append(model._opacity.detach().cpu().numpy())
            all_scales.append(model._scaling.detach().cpu().numpy())
            all_rots.append(model._rotation.detach().cpu().numpy())
            all_semantics.append(model._semantic.detach().cpu().numpy())
            all_model_ids.append(np.full(num_points, model_id, dtype=np.int32))
            if model_name == 'background':
                all_main_directions.append(np.tile(model.main_direction.detach().cpu().numpy(), (num_points, 1)))
                all_init_pos.append(np.tile(model.init_pos.detach().cpu().numpy(), (num_points, 1)))
                bounds = []
                for b in model.bounds:
                    if torch.is_tensor(b):
                        b = b.detach().cpu().numpy()        #  NumPy
                        if b.size == 1:
                            b = b.item()
                        else:
                            b = b.tolist()                  #  Python
                    bounds.append(b)
                #  NumPy , flatten,tile
                bounds = np.array(bounds, dtype=np.float32).flatten()
                bounds = bounds.reshape(1, -1)
                bounds = np.tile(bounds, (num_points, 1))
                all_bounds.append(bounds)
            else:
                all_main_directions.append(np.zeros((num_points, 3), dtype=np.float32))
                all_init_pos.append(np.zeros((num_points, 3), dtype=np.float32))
                if all_bounds:
                    bounds_shape = all_bounds[0].shape[1]  #  bounds
                else:
                    bounds_shape = 6
                all_bounds.append(np.zeros((num_points, bounds_shape), dtype=np.float32))


        anchor = np.concatenate(all_xyz, axis=0)
        levels = np.concatenate(all_levels, axis=0)
        extra_levels = np.concatenate(all_extra_levels, axis=0)
        anchor_feats = np.concatenate(all_anchor_feats, axis=0)
        offsets = np.concatenate(all_offsets, axis=0)
        opacities = np.concatenate(all_opacities, axis=0)
        scales = np.concatenate(all_scales, axis=0)
        rots = np.concatenate(all_rots, axis=0)
        semantic = np.concatenate(all_semantics, axis=0)
        model_ids = np.concatenate(all_model_ids, axis=0)

        if all_main_directions:
            main_direction = np.concatenate(all_main_directions, axis=0)
            init_pos = np.concatenate(all_init_pos, axis=0)
            bounds = np.concatenate(all_bounds, axis=0)
        else:
            main_direction = np.empty((0, 3))
            init_pos = np.empty((0, 3))
            bounds = np.empty((0, 2))

        # infos
        infos = np.zeros((anchor.shape[0], 2), dtype=np.float32)
        # voxel_sizestandard_dist
        first_model = getattr(self, list(self.model_name_id.keys())[0])

        voxel_size_val = first_model.voxel_size
        if torch.is_tensor(voxel_size_val):
            voxel_size_val = voxel_size_val.cpu().numpy()
        if isinstance(voxel_size_val, np.ndarray) and voxel_size_val.size > 1:
            voxel_size_val = float(np.mean(voxel_size_val))
        else:
            voxel_size_val = float(voxel_size_val.item() if hasattr(voxel_size_val, 'item') else voxel_size_val)
        infos[:, 0] = voxel_size_val

        # standard_dist:
        standard_dist_val = first_model.standard_dist
        if torch.is_tensor(standard_dist_val):
            standard_dist_val = standard_dist_val.cpu().numpy()
        if isinstance(standard_dist_val, np.ndarray) and standard_dist_val.size > 1:
            standard_dist_val = float(np.mean(standard_dist_val))
        else:
            standard_dist_val = float(standard_dist_val.item() if hasattr(standard_dist_val, 'item') else standard_dist_val)
        infos[:, 1] = standard_dist_val


        dtype_full = []
        dtype_full.extend([('x', 'f4'), ('y', 'f4'), ('z', 'f4')])  # xyz coordinates
        dtype_full.append(('level', 'f4'))  # level
        dtype_full.append(('extra_level', 'f4'))  # extra_level
        dtype_full.extend([('info_0', 'f4'), ('info_1', 'f4')])  # info
        dtype_full.append(('model_id', 'i4'))

        # offsets
        n_offsets = offsets.shape[1]
        for i in range(n_offsets):
            dtype_full.append((f'f_offset_{i}', 'f4'))

        # anchor features
        n_features = anchor_feats.shape[1]
        for i in range(n_features):
            dtype_full.append((f'f_anchor_feat_{i}', 'f4'))

        # opacity
        dtype_full.append(('opacity', 'f4'))

        # scales
        n_scales = scales.shape[1]
        for i in range(n_scales):
            dtype_full.append((f'scale_{i}', 'f4'))

        # rotations
        n_rots = rots.shape[1]
        for i in range(n_rots):
            dtype_full.append((f'rot_{i}', 'f4'))

        # semantic
        n_semantic = semantic.shape[1]
        for i in range(n_semantic):
            dtype_full.append((f'semantic_{i}', 'f4'))

        n_main_direction = main_direction.shape[1]
        for i in range(n_main_direction):
            dtype_full.append((f'main_direction_{i}', 'f4'))
        n_init_pos = init_pos.shape[1]
        for i in range(n_init_pos):
            dtype_full.append((f'init_pos_{i}', 'f4'))
        bounds_array = np.array(bounds)
        for i in range(bounds_array.shape[1]):
            dtype_full.append((f'bounds_{i}', 'f4'))

        elements = np.empty(anchor.shape[0], dtype=dtype_full)

        arrays = [anchor, levels, extra_levels, infos, model_ids[..., np.newaxis], offsets, anchor_feats,
          opacities, scales, rots, semantic,
          main_direction, init_pos, bounds]


        processed = []
        for arr in arrays:
            if arr.ndim == 1:

                arr = arr.reshape(-1, 1)
            processed.append(arr)

        attributes = np.concatenate(processed, axis=1)


        assert attributes.shape[1] == len(dtype_full), f"Attribute count mismatch: got {attributes.shape[1]}, expected {len(dtype_full)}"


        elements[:] = list(map(tuple, attributes))

        # PLY
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path):
        print(f"Loading PLY data from {path}")
        plydata = PlyData.read(path)['vertex']


        xyz = np.stack((np.asarray(plydata['x']),
                       np.asarray(plydata['y']),
                       np.asarray(plydata['z'])), axis=1).astype(np.float32)
        total_points = xyz.shape[0]
        print(f"Loaded {total_points} points")


        model_ids = np.asarray(plydata['model_id'])


        model_sizes = {}
        for model_name, model_id in self.model_name_id.items():
            model_sizes[model_name] = np.sum(model_ids == model_id)
            print(f"Model {model_name} has {model_sizes[model_name]} points")


        levels = np.asarray(plydata['level'])[..., np.newaxis].astype(np.int32)
        extra_levels = np.asarray(plydata['extra_level'])[..., np.newaxis].astype(np.float32)

        #  info
        self.voxel_size = torch.tensor(plydata['info_0'][0]).float()
        self.standard_dist = torch.tensor(plydata['info_1'][0]).float()

        #  opacity
        opacities = np.asarray(plydata['opacity'])[..., np.newaxis].astype(np.float32)

        #  scales
        scale_names = [p.name for p in plydata.properties if p.name.startswith('scale_')]
        scale_names = sorted(scale_names, key=lambda x: int(x.split('_')[-1]))
        scales = np.zeros((xyz.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata[attr_name]).astype(np.float32)

        #  rotations
        rot_names = [p.name for p in plydata.properties if p.name.startswith('rot')]
        rot_names = sorted(rot_names, key=lambda x: int(x.split('_')[-1]))
        rots = np.zeros((xyz.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata[attr_name]).astype(np.float32)

        #  anchor features
        anchor_feat_names = [p.name for p in plydata.properties if p.name.startswith('f_anchor_feat')]
        anchor_feat_names = sorted(anchor_feat_names, key=lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((xyz.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata[attr_name]).astype(np.float32)

        #  offsets
        offset_names = [p.name for p in plydata.properties if p.name.startswith('f_offset')]
        offset_names = sorted(offset_names, key=lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((xyz.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata[attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        #  semantic
        semantic_names = [p.name for p in plydata.properties if p.name.startswith('semantic')]
        semantic_names = sorted(semantic_names, key=lambda x: int(x.split('_')[-1]))
        semantic = np.zeros((xyz.shape[0], len(semantic_names)))
        for idx, attr_name in enumerate(semantic_names):
            semantic[:, idx] = np.asarray(plydata[attr_name]).astype(np.float32)

        main_direction = np.stack((np.asarray(plydata["main_direction_0"]),
                                   np.asarray(plydata["main_direction_1"]),
                                   np.asarray(plydata["main_direction_2"])), axis=1).astype(np.float32)
        init_pos = np.stack((np.asarray(plydata["init_pos_0"]),
                             np.asarray(plydata["init_pos_1"]),
                             np.asarray(plydata["init_pos_2"])), axis=1).astype(np.float32)
        bounds = np.stack((np.asarray(plydata["bounds_0"]),
                           np.asarray(plydata["bounds_1"])), axis=1).astype(np.float32)
         #  main_direction  init_pos
        if main_direction.shape[0] > 1 and np.all(main_direction == main_direction[0]):
            main_direction = main_direction[0]
        if init_pos.shape[0] > 1 and np.all(init_pos == init_pos[0]):
            init_pos = init_pos[0]


        for model_name, model_id in self.model_name_id.items():
            model = getattr(self, model_name)
            mask = (model_ids == model_id)

            # tensor
            model._anchor = nn.Parameter(torch.tensor(xyz[mask], dtype=torch.float, device="cuda").requires_grad_(True))
            model._level = torch.tensor(levels[mask], dtype=torch.int, device="cuda")
            model._extra_level = torch.tensor(extra_levels[mask], dtype=torch.float, device="cuda").squeeze(dim=1)
            model._offset = nn.Parameter(torch.tensor(offsets[mask], dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
            model._anchor_feat = nn.Parameter(torch.tensor(anchor_feats[mask], dtype=torch.float, device="cuda").requires_grad_(True))
            model._scaling = nn.Parameter(torch.tensor(scales[mask], dtype=torch.float, device="cuda").requires_grad_(True))
            model._opacity = nn.Parameter(torch.tensor(opacities[mask], dtype=torch.float, device="cuda").requires_grad_(False))
            model._semantic = nn.Parameter(torch.tensor(semantic[mask], dtype=torch.float, device="cuda").requires_grad_(True))
            model._rotation = nn.Parameter(torch.tensor(rots[mask], dtype=torch.float, device="cuda").requires_grad_(False))
            model._anchor_mask = torch.ones(model._anchor.shape[0], dtype=torch.bool, device="cuda")
            model.voxel_size = self.voxel_size
            model.standard_dist = self.standard_dist

            #  levels
            if model._level.numel() > 0:
                model.levels = torch.max(model._level) - torch.min(model._level) + 1
            else:
                print(f"Warning: Model {model_name} has no points, setting levels to 1")
                model.levels = torch.tensor(1)

            if model_name == "background":
                model.main_direction = torch.tensor(main_direction[0], dtype=torch.float, device="cuda")
                model.init_pos = torch.tensor(init_pos[0], dtype=torch.float, device="cuda")
                model.bounds = bounds[0].tolist()
            elif hasattr(model, "_anchor") and model._anchor.numel() > 0:
                # Actor anchor growing needs the same lattice origin created during
                # fresh initialization. Older aggregate PLY files store zero actor
                # init_pos, so fall back to the loaded actor anchors.
                model_init_pos = None
                if isinstance(init_pos, np.ndarray) and init_pos.ndim == 2 and mask.any():
                    per_model_init = init_pos[mask]
                    if per_model_init.size > 0 and not np.allclose(per_model_init, 0):
                        model_init_pos = per_model_init[0]
                if model_init_pos is None:
                    anchor_min = float(model._anchor.detach().min().item())
                    model_init_pos = np.array([anchor_min, anchor_min, anchor_min], dtype=np.float32)
                model.init_pos = torch.tensor(model_init_pos, dtype=torch.float, device="cuda")

        self.active_sh_degree = self.max_sh_degree

    def replace_tensor_to_optimizer(self, tensor, name):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if group["name"] == name:
                stored_state = self.optimizer.state.get(group['params'][0], None)
                stored_state["exp_avg"] = torch.zeros_like(tensor)
                stored_state["exp_avg_sq"] = torch.zeros_like(tensor)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(tensor.requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
        return optimizable_tensors


    def cat_tensors_to_optimizer(self, tensors_dict):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue
            if len(group["params"])>1:continue
            assert len(group["params"]) == 1
            extension_tensor = tensors_dict[group["name"]]
            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = torch.cat((stored_state["exp_avg"], torch.zeros_like(extension_tensor)), dim=0)
                stored_state["exp_avg_sq"] = torch.cat((stored_state["exp_avg_sq"], torch.zeros_like(extension_tensor)), dim=0)

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                self.optimizer.state[group['params'][0]] = stored_state

                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(torch.cat((group["params"][0], extension_tensor), dim=0).requires_grad_(True))
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors


    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask,
                        anchor_visible_mask, total_ids, opacity_ids):
        viewspace_grad = viewspace_point_tensor.grad

        valid_models = []
        for model_name in self.graph_gaussian_range.keys():
            if len(valid_models) < len(opacity_ids):
                valid_models.append(model_name)
            else:
                break

        idx = 0
        graph_offset_range = {}
        for num, model_name in enumerate(valid_models):
            num_gaussian = opacity_ids[num].shape[0]
            graph_offset_range[model_name] = [idx, idx + num_gaussian - 1]
            idx += num_gaussian

        idx = 0
        graph_point_range = {}
        for num, model_name in enumerate(valid_models):
            if num < len(total_ids):  # total_ids
                num_gaussian = total_ids[num].shape[0]
                graph_point_range[model_name] = [idx, idx + num_gaussian - 1]
                idx += num_gaussian
            else:
                continue


        for model_name in valid_models:
            if model_name not in graph_point_range or model_name not in graph_offset_range:
                continue

            model: GaussianModel = getattr(self, model_name)

            start2, end2 = graph_point_range[model_name]
            end2 = end2 + 1
            visibility_model = update_filter[start2:end2]
            viewpoint_grad = viewspace_grad[start2:end2]

            start1, end1 = graph_offset_range[model_name]
            end1 = end1 + 1
            offset_select = offset_selection_mask[start1:end1]
            single_opacity = opacity[start1:end1]

            start, end = self.graph_gaussian_range[model_name]
            end = end + 1
            visible_mask = anchor_visible_mask[start:end]
            model.training_statis(viewpoint_grad, single_opacity, visibility_model, offset_select, visible_mask)

    def prune_anchor(self,mask):
        valid_points_mask = ~mask

        optimizable_tensors = self._prune_anchor_optimizer(valid_points_mask)

        self._anchor = optimizable_tensors["anchor"]
        self._offset = optimizable_tensors["offset"]
        self._anchor_feat = optimizable_tensors["anchor_feat"]
        self._opacity = optimizable_tensors["opacity"]
        self._scaling = optimizable_tensors["scaling"]
        self._rotation = optimizable_tensors["rotation"]
        self._semantic = optimizable_tensors["semantic"]
        self._level = self._level[valid_points_mask]
        self._extra_level = self._extra_level[valid_points_mask]

    def get_remove_duplicates(self, grid_coords, selected_grid_coords_unique, use_chunk = True):
        for model_name in self.graph_gaussian_range.keys():
            model: GaussianModel = getattr(self, model_name)
            start, end = self.graph_gaussian_range[model_name]
            end = end + 1
            if use_chunk:
                model.get_remove_duplicates(grid_coords, selected_grid_coords_unique, use_chunk)
            else:
                model.get_remove_duplicates(grid_coords, selected_grid_coords_unique, use_chunk)
    def adjust_anchor(self, iteration, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, update_ratio=0.5, extra_ratio=4.0, extra_up=0.25, min_opacity=0.005):
        for model_name in self.graph_gaussian_range.keys():
            model: GaussianModel = getattr(self, model_name)
            if model_name == 'background':
                model.adjust_anchor(iteration, check_interval, success_threshold, grad_threshold, update_ratio, extra_ratio, extra_up, min_opacity)
            else:
                model.adjust_anchor(iteration, check_interval, success_threshold, grad_threshold, update_ratio,
                                    extra_ratio, extra_up, min_opacity, self.viewpoint_camera)


    def save_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            self.eval()

            # MLP
            opacity_params = []
            cov_params = []
            color_params = []
            feature_bank_params = []
            appearance_params = []
            hash_encoding_params = []
            hash_proj_params = []
            hash_lod_partitioner_params = {}

            for model_name in self.model_name_id.keys():
                model: GaussianModel = getattr(self, model_name)

                # MLP
                opacity_params.append(model.mlp_opacity.state_dict())
                cov_params.append(model.mlp_cov.state_dict())
                color_params.append(model.mlp_color.state_dict())

                if model.use_feat_bank:
                    feature_bank_params.append(model.mlp_feature_bank.state_dict())

                if model.appearance_dim > 0:
                    appearance_params.append(model.embedding_appearance.state_dict())

                if model.use_hash_encoding and hasattr(model, 'hash_encoding') and model.hash_encoding is not None:
                    hash_encoding_params.append(model.hash_encoding.state_dict())

                if hasattr(model, 'hash_feat_multi_proj') and model.hash_feat_multi_proj is not None:
                    hash_proj_params.append(model.hash_feat_multi_proj.state_dict())

                if hasattr(model, 'hash_lod_partitioner') and model.hash_lod_partitioner is not None:
                    hash_lod_partitioner_params[model_name] = model.hash_lod_partitioner.save_state()

            # MLP
            torch.save(opacity_params, os.path.join(path, 'opacity_mlp.pt'))
            torch.save(cov_params, os.path.join(path, 'cov_mlp.pt'))
            torch.save(color_params, os.path.join(path, 'color_mlp.pt'))

            if feature_bank_params:
                torch.save(feature_bank_params, os.path.join(path, 'feature_bank_mlp.pt'))
            if appearance_params:
                torch.save(appearance_params, os.path.join(path, 'appearance.pt'))
            if hash_encoding_params:
                torch.save(hash_encoding_params, os.path.join(path, 'hash_encoding.pt'))
                print(f" Saved hash_encoding parameters for {len(hash_encoding_params)} models")
            if hash_proj_params:
                torch.save(hash_proj_params, os.path.join(path, 'hash_feat_multi_proj.pt'))
                print(f" Saved hash_feat_multi_proj parameters for {len(hash_proj_params)} models")
            if hash_lod_partitioner_params:
                torch.save(hash_lod_partitioner_params, os.path.join(path, 'hash_lod_partitioner.pt'))
                print(f" Saved hash_lod_partitioner for {len(hash_lod_partitioner_params)} models: {list(hash_lod_partitioner_params.keys())}")

            self.train()

        elif mode == 'unite':
            param_dict = {}

            #  MLP
            opacity_params = []
            cov_params = []
            color_params = []
            feature_bank_params = []
            appearance_params = []
            hash_encoding_params = []
            hash_proj_params = []
            hash_lod_partitioner_params = {}

            for model_name in self.model_name_id.keys():
                model: GaussianModel = getattr(self, model_name)

                #  MLP
                opacity_params.append(model.mlp_opacity.state_dict())
                cov_params.append(model.mlp_cov.state_dict())
                color_params.append(model.mlp_color.state_dict())

                if model.use_feat_bank:
                    feature_bank_params.append(model.mlp_feature_bank.state_dict())

                if model.appearance_dim > 0:
                    appearance_params.append(model.embedding_appearance.state_dict())

                if model.use_hash_encoding and hasattr(model, 'hash_encoding') and model.hash_encoding is not None:
                    hash_encoding_params.append(model.hash_encoding.state_dict())

                if hasattr(model, 'hash_feat_multi_proj') and model.hash_feat_multi_proj is not None:
                    hash_proj_params.append(model.hash_feat_multi_proj.state_dict())

                if hasattr(model, 'hash_lod_partitioner') and model.hash_lod_partitioner is not None:
                    hash_lod_partitioner_params[model_name] = model.hash_lod_partitioner.save_state()



            # MLP
            torch.save(opacity_params, os.path.join(path, 'opacity_mlp.pt'))
            torch.save(cov_params, os.path.join(path, 'cov_mlp.pt'))
            torch.save(color_params, os.path.join(path, 'color_mlp.pt'))

            if feature_bank_params:
                torch.save(feature_bank_params, os.path.join(path, 'feature_bank_mlp.pt'))
            if appearance_params:
                torch.save(appearance_params, os.path.join(path, 'appearance.pt'))
            if hash_encoding_params:
                torch.save(hash_encoding_params, os.path.join(path, 'hash_encoding.pt'))
                print(f" Saved hash_encoding parameters for {len(hash_encoding_params)} models")
            if hash_proj_params:
                torch.save(hash_proj_params, os.path.join(path, 'hash_feat_multi_proj.pt'))
                print(f" Saved hash_feat_multi_proj parameters for {len(hash_proj_params)} models")
            if hash_lod_partitioner_params:
                torch.save(hash_lod_partitioner_params, os.path.join(path, 'hash_lod_partitioner.pt'))
                print(f" Saved hash_lod_partitioner for {len(hash_lod_partitioner_params)} models: {list(hash_lod_partitioner_params.keys())}")
        else:
            raise NotImplementedError

    def load_mlp_checkpoints(self, path, mode = 'split'):#split or unite
        if mode == 'split':

            # MLP
            opacity_params = torch.load(os.path.join(path, 'opacity_mlp.pt'))
            cov_params = torch.load(os.path.join(path, 'cov_mlp.pt'))
            color_params = torch.load(os.path.join(path, 'color_mlp.pt'))

            feature_bank_path = os.path.join(path, 'feature_bank_mlp.pt')
            if os.path.exists(feature_bank_path):
                feature_bank_params = torch.load(feature_bank_path)

            appearance_path = os.path.join(path, 'appearance.pt')
            if os.path.exists(appearance_path):
                appearance_params = torch.load(appearance_path)

            hash_encoding_path = os.path.join(path, 'hash_encoding.pt')
            hash_encoding_params = None
            if os.path.exists(hash_encoding_path):
                hash_encoding_params = torch.load(hash_encoding_path)
                print(f" [Split Mode] Loaded hash_encoding for {len(hash_encoding_params)} models")

            hash_proj_path = os.path.join(path, 'hash_feat_multi_proj.pt')
            hash_proj_params = None
            if os.path.exists(hash_proj_path):
                hash_proj_params = torch.load(hash_proj_path)
                print(f" [Split Mode] Loaded hash_feat_multi_proj for {len(hash_proj_params)} models")

            hash_lod_partitioner_path = os.path.join(path, 'hash_lod_partitioner.pt')
            hash_lod_partitioner_params = None
            if os.path.exists(hash_lod_partitioner_path):
                hash_lod_partitioner_params = torch.load(hash_lod_partitioner_path)
                print(f" [Split Mode] Loaded hash_lod_partitioner for {len(hash_lod_partitioner_params)} models: {list(hash_lod_partitioner_params.keys())}")

            for i, model_name in enumerate(self.model_name_id.keys()):
                model: GaussianModel = getattr(self, model_name)

                model.mlp_opacity.load_state_dict(opacity_params[i])
                model.mlp_cov.load_state_dict(cov_params[i])
                model.mlp_color.load_state_dict(color_params[i])

                if model.use_feat_bank and os.path.exists(feature_bank_path):
                    model.mlp_feature_bank.load_state_dict(feature_bank_params[i])

                if model.appearance_dim > 0 and os.path.exists(appearance_path):
                    model.embedding_appearance.load_state_dict(appearance_params[i])

                if model.use_hash_encoding and hash_encoding_params is not None and i < len(hash_encoding_params):
                    saved_state = hash_encoding_params[i]
                    #  "tables.0.weight", "tables.1.weight", ...
                    saved_n_levels = 0
                    for key in saved_state.keys():
                        if key.startswith("tables.") and key.endswith(".weight"):
                            level_idx = int(key.split(".")[1])
                            saved_n_levels = max(saved_n_levels, level_idx + 1)
                    if saved_n_levels == 0:
                        saved_n_levels = getattr(model, 'hash_levels', 8) or 8

                    if model.hash_encoding is None:
                        if hasattr(model, '_anchor') and model._anchor is not None and model._anchor.shape[0] > 0:
                            xyz_min = torch.min(model._anchor.detach(), dim=0).values
                            xyz_max = torch.max(model._anchor.detach(), dim=0).values
                            #   checkpoint
                            model.init_hash_encoding(xyz_min, xyz_max, n_levels=saved_n_levels)
                            print(f"[{model_name}]  Created hash_encoding with {saved_n_levels} levels before loading")

                    if model.hash_encoding is not None:
                        model.hash_encoding.load_state_dict(saved_state)
                        print(f"[{model_name}]  Loaded hash_encoding from checkpoint")
                    else:
                        print(f"[{model_name}]   Could not create hash_encoding!")
                elif model.use_hash_encoding:
                    print(f"[{model_name}]   hash_encoding params not in checkpoint!")

                if hash_proj_params is not None and i < len(hash_proj_params):
                    if not hasattr(model, 'hash_feat_multi_proj') or model.hash_feat_multi_proj is None:
                        saved_state = hash_proj_params[i]
                        if '0.weight' in saved_state:
                            in_dim = saved_state['0.weight'].shape[1]
                            out_dim = saved_state['2.weight'].shape[0] if '2.weight' in saved_state else saved_state['0.weight'].shape[0]
                            import torch.nn as nn
                            model.hash_feat_multi_proj = nn.Sequential(
                                nn.Linear(in_dim, out_dim),
                                nn.ReLU(),
                                nn.Linear(out_dim, out_dim)
                            ).cuda()
                            print(f"[{model_name}]  Created hash_feat_multi_proj: {in_dim} -> {out_dim}")

                    if model.hash_feat_multi_proj is not None:
                        model.hash_feat_multi_proj.load_state_dict(hash_proj_params[i])
                        print(f"[{model_name}]  Loaded hash_feat_multi_proj from checkpoint")

                if hash_lod_partitioner_params is not None and model_name in hash_lod_partitioner_params:
                    from scene.hash_lod_partitioner import HashLODPartitioner
                    # partitioner,
                    if not hasattr(model, 'hash_lod_partitioner') or model.hash_lod_partitioner is None:
                        n_lod_levels = getattr(model, 'levels', 8)
                        n_hash_levels = int(getattr(model, 'hash_levels', 12)) if getattr(model, 'hash_levels', None) else 12
                        base_voxel_size = float(model.voxel_size.max().item() if isinstance(model.voxel_size, torch.Tensor) else model.voxel_size)
                        model.hash_lod_partitioner = HashLODPartitioner(
                            n_lod_levels=n_lod_levels,
                            n_hash_levels=n_hash_levels,
                            hash_features_per_level=int(getattr(model, 'hash_feat_dim', 2)),
                            base_voxel_size=base_voxel_size,
                            fork=float(getattr(model, 'fork', 2.0)),
                            use_learnable_lod_bias=getattr(model, 'use_lod_bias', True),
                            device='cuda'
                        )
                        print(f"[{model_name}]  Created hash_lod_partitioner before loading")

                    model.hash_lod_partitioner.load_state(hash_lod_partitioner_params[model_name])
                    model.use_hash_lod_partitioner = True
                    if hasattr(model.hash_lod_partitioner, 'bounds'):
                        model.bounds = model.hash_lod_partitioner.bounds
                    if hasattr(model.hash_lod_partitioner, 'main_direction') and model.hash_lod_partitioner.main_direction is not None:
                        model.main_direction = model.hash_lod_partitioner.main_direction
                    if hasattr(model.hash_lod_partitioner, 'near_is_low'):
                        model.near_is_low = model.hash_lod_partitioner.near_is_low
                    #  dynamic_voxel_sizes
                    if hasattr(model, 'voxel_size'):
                        base_voxel = float(model.voxel_size.max().item() if isinstance(model.voxel_size, torch.Tensor) else model.voxel_size)
                        model.dynamic_voxel_sizes = {
                            "near": model.hash_lod_partitioner.region_voxel_scales["near"].item() * base_voxel,
                            "mid": model.hash_lod_partitioner.region_voxel_scales["mid"].item() * base_voxel,
                            "far": model.hash_lod_partitioner.region_voxel_scales["far"].item() * base_voxel,
                        }
                    if model.hash_lod_partitioner._lod_bias is not None:
                        model._lod_bias = model.hash_lod_partitioner._lod_bias
                        print(f"[{model_name}]  Synced _lod_bias from hash_lod_partitioner (size={model._lod_bias.shape[0]})")
                    if model.hash_lod_partitioner._opacity_scale is not None:
                        model._opacity_scale = model.hash_lod_partitioner._opacity_scale
                        print(f"[{model_name}]  Synced _opacity_scale from hash_lod_partitioner (size={model._opacity_scale.shape[0]})")
                    print(f"[{model_name}]  Loaded hash_lod_partitioner from checkpoint (bounds={model.bounds}, near_is_low={model.near_is_low})")

        elif mode == 'unite':

            # MLP
            opacity_params = torch.load(os.path.join(path, 'opacity_mlp.pt'))
            cov_params = torch.load(os.path.join(path, 'cov_mlp.pt'))
            color_params = torch.load(os.path.join(path, 'color_mlp.pt'))

            feature_bank_path = os.path.join(path, 'feature_bank_mlp.pt')
            if os.path.exists(feature_bank_path):
                feature_bank_params = torch.load(feature_bank_path)

            appearance_path = os.path.join(path, 'appearance.pt')
            if os.path.exists(appearance_path):
                appearance_params = torch.load(appearance_path)

            hash_encoding_path = os.path.join(path, 'hash_encoding.pt')
            hash_encoding_params = None
            if os.path.exists(hash_encoding_path):
                hash_encoding_params = torch.load(hash_encoding_path)
                print(f" [Unite Mode] Loaded hash_encoding for {len(hash_encoding_params)} models")

            hash_proj_path = os.path.join(path, 'hash_feat_multi_proj.pt')
            hash_proj_params = None
            if os.path.exists(hash_proj_path):
                hash_proj_params = torch.load(hash_proj_path)
                print(f" [Unite Mode] Loaded hash_feat_multi_proj for {len(hash_proj_params)} models")

            for i, model_name in enumerate(self.model_name_id.keys()):
                model: GaussianModel = getattr(self, model_name)

                model.mlp_opacity.load_state_dict(opacity_params[i])
                model.mlp_cov.load_state_dict(cov_params[i])
                model.mlp_color.load_state_dict(color_params[i])

                if model.use_feat_bank and os.path.exists(feature_bank_path):
                    model.mlp_feature_bank.load_state_dict(feature_bank_params[i])

                if model.appearance_dim > 0 and os.path.exists(appearance_path):
                    model.embedding_appearance.load_state_dict(appearance_params[i])

                if model.use_hash_encoding and hash_encoding_params is not None and i < len(hash_encoding_params):
                    saved_state = hash_encoding_params[i]
                    saved_n_levels = 0
                    for key in saved_state.keys():
                        if key.startswith("tables.") and key.endswith(".weight"):
                            level_idx = int(key.split(".")[1])
                            saved_n_levels = max(saved_n_levels, level_idx + 1)
                    if saved_n_levels == 0:
                        saved_n_levels = getattr(model, 'hash_levels', 8) or 8

                    if model.hash_encoding is None:
                        if hasattr(model, '_anchor') and model._anchor is not None and model._anchor.shape[0] > 0:
                            xyz_min = torch.min(model._anchor.detach(), dim=0).values
                            xyz_max = torch.max(model._anchor.detach(), dim=0).values
                            #   checkpoint
                            model.init_hash_encoding(xyz_min, xyz_max, n_levels=saved_n_levels)
                            print(f"[{model_name}]  Created hash_encoding with {saved_n_levels} levels before loading (unite)")

                    if model.hash_encoding is not None:
                        model.hash_encoding.load_state_dict(saved_state)
                        print(f"[{model_name}]  Loaded hash_encoding (unite mode)")
                    else:
                        print(f"[{model_name}]   Could not create hash_encoding (unite)!")
                elif model.use_hash_encoding:
                    print(f"[{model_name}]   hash_encoding params not found (unite)!")

                if hash_proj_params is not None and i < len(hash_proj_params):
                    if not hasattr(model, 'hash_feat_multi_proj') or model.hash_feat_multi_proj is None:
                        saved_state = hash_proj_params[i]
                        if '0.weight' in saved_state:
                            in_dim = saved_state['0.weight'].shape[1]
                            out_dim = saved_state['2.weight'].shape[0] if '2.weight' in saved_state else saved_state['0.weight'].shape[0]
                            import torch.nn as nn
                            model.hash_feat_multi_proj = nn.Sequential(
                                nn.Linear(in_dim, out_dim),
                                nn.ReLU(),
                                nn.Linear(out_dim, out_dim)
                            ).cuda()
                            print(f"[{model_name}]  Created hash_feat_multi_proj (unite): {in_dim} -> {out_dim}")

                    if model.hash_feat_multi_proj is not None:
                        model.hash_feat_multi_proj.load_state_dict(hash_proj_params[i])
                        print(f"[{model_name}]  Loaded hash_feat_multi_proj (unite mode)")

        else:
            raise NotImplementedError
    @property
    def get_normal(self):
        normals = []
        if self.get_visibility('background'):
            normals_bkgd = self.background.get_normals()
            normals.append(normals_bkgd)

        for i, obj_name in enumerate(self.graph_obj_list):
            obj_model: GaussianModelActor = getattr(self, obj_name)
            track_id = obj_model.track_id

            normals_obj_local = obj_model.get_normals()

            obj_rot = self.actor_pose.get_tracking_rotation(track_id, self.viewpoint_camera)
            obj_rot = quaternion_to_matrix(obj_rot.unsqueeze(0)).squeeze(0)

            normals_obj_global = normals_obj_local @ obj_rot.T
            normals_obj_global = torch.nn.functinal.normalize(normals_obj_global)
            normals.append(normals_obj_global)

        normals = torch.cat(normals, dim=0)
        return normals

    @property
    def get_semantic(self):
        semantic_collect = []
        if self.get_visibility('background'):
            semantic_bkgd = self.background.get_semantic()
            semantic_collect.append(semantic_bkgd)
        for i, obj_name in enumerate(self.graph_obj_list):
            obj_model: GaussianModelActor = getattr(self, obj_name)
            semantic_obj = obj_model.get_semantic()
            semantic_collect.append(semantic_obj)
        semantic = torch.cat(semantic_collect, dim=0)
        self._semantic = semantic
        return semantic

    def update_obj_center_cache(self):
        """(densification)"""
        if not hasattr(self, '_obj_center_cache'):
            self._obj_center_cache = {}
            self._obj_extent_cache = {}

        for obj_name in self.obj_list:
            obj_model: GaussianModelActor = getattr(self, obj_name)
            if hasattr(obj_model, '_anchor') and obj_model._anchor.shape[0] > 0:
                self._obj_center_cache[obj_name] = obj_model._anchor.mean(dim=0).detach()
                anchor_std = obj_model._anchor.std(dim=0)
                self._obj_extent_cache[obj_name] = anchor_std.max().item() * 3.0

    def parse_camera(self, cameras):
        # Set camera and background mask
        self.viewpoint_camera = cameras
        self.num_gaussians = 0

        # Cache visibility checks
        background_visible = self.get_visibility('background')
        # Background section
        if background_visible:
            num_gaussians_bkgd = self.background.get_anchor.shape[0]
            self.num_gaussians += num_gaussians_bkgd

        # Object section (build scene graph)
        self.graph_obj_list = []

        if self.include_obj:
            timestamp = cameras.meta.get('timestamp', cameras.time) if hasattr(cameras, 'meta') else cameras.time
            for obj_name in self.obj_list:
                obj_model: GaussianModelActor = getattr(self, obj_name)
                # Cache start and end timestamps
                start_timestamp, end_timestamp = obj_model.start_timestamp, obj_model.end_timestamp
                if start_timestamp <= timestamp <= end_timestamp and self.get_visibility(obj_name):
                    self.graph_obj_list.append(obj_name)
                    self.num_gaussians += obj_model._anchor.shape[0]

        # Set index range
        self.graph_gaussian_range = dict()
        idx = 0

        if background_visible:
            self.graph_gaussian_range['background'] = [idx, idx + num_gaussians_bkgd - 1]
            idx += num_gaussians_bkgd

        for obj_name in self.graph_obj_list:
            obj_model: GaussianModelActor = getattr(self, obj_name)
            num_gaussians_obj = obj_model._anchor.shape[0]
            self.graph_gaussian_range[obj_name] = [idx, idx + num_gaussians_obj - 1]
            idx += num_gaussians_obj

        # Rotation and translation handling
        self.obj_rots = []
        self.obj_trans = []

        # Early exit if no objects are included
        if not self.graph_obj_list:
            self.obj_rots = torch.empty((0, 4), dtype=torch.float32, device='cuda')
            self.obj_trans = torch.empty((0, 3), dtype=torch.float32, device='cuda')
            return
        axis_transform = cameras.axis_transform
        ego_pose = self.viewpoint_camera.ego_pose
        axis_t_rot = matrix_to_quaternion(axis_transform[:3, :3].unsqueeze(0)).squeeze(0)   # added
        ego_pose_rot = matrix_to_quaternion(ego_pose[:3, :3].unsqueeze(0)).squeeze(0)
        ego_pose_rot = quaternion_raw_multiply(axis_t_rot.unsqueeze(0), ego_pose_rot.unsqueeze(0)).squeeze(0)   # added
        axis_transform_matrix = axis_transform[:3, :3]
        ego_pose_matrix = ego_pose[:3, :3]
        ego_pose_translation = ego_pose[:3, 3]
        ego_pose_matrix = axis_transform_matrix @ ego_pose_matrix    # added
        ego_pose_translation = axis_transform_matrix @ ego_pose_translation + axis_transform[:3, 3]

        # Use torch.no_grad() if gradients are not required
        with torch.no_grad():
            for obj_name in self.graph_obj_list:
                obj_model: GaussianModelActor = getattr(self, obj_name)
                track_id = obj_model.track_id

                # Get rotation and translation from actor_pose
                obj_rot = self.actor_pose.get_tracking_rotation(track_id, self.viewpoint_camera)
                obj_trans = self.actor_pose.get_tracking_translation(track_id, self.viewpoint_camera)

                # Combine ego_pose rotation and translation
                obj_rot = quaternion_raw_multiply(ego_pose_rot.unsqueeze(0), obj_rot.unsqueeze(0)).squeeze(0)
                obj_trans = ego_pose_matrix @ obj_trans + ego_pose_translation

                # Expand once outside the main loop for efficiency
                num_anchors = obj_model._anchor.shape[0]
                # Skip objects with zero anchors
                if num_anchors == 0:
                    continue

                obj_rot_expanded = obj_rot.expand(num_anchors, -1)
                obj_trans_expanded = obj_trans.unsqueeze(0).expand(num_anchors, -1)

                # Assign the expanded rotations and translations
                obj_model.obj_rots = obj_rot_expanded
                obj_model.obj_trans = obj_trans_expanded

                # Append results to lists
                self.obj_rots.append(obj_rot_expanded)
                self.obj_trans.append(obj_trans_expanded)

            # Concatenate all object rotations and translations outside the loop
            if len(self.obj_rots) > 0:
                self.obj_rots = torch.cat(self.obj_rots, dim=0)
                self.obj_trans = torch.cat(self.obj_trans, dim=0)
            else:
                self.obj_rots = torch.empty((0, 4), dtype=torch.float32, device='cuda')
                self.obj_trans = torch.empty((0, 3), dtype=torch.float32, device='cuda')

    def set_visibility(self, include_list):
        self.include_list = include_list # prefix

    def get_visibility(self, model_name):
        if model_name == 'background':
            if model_name in self.include_list and self.include_background:
                return True
            else:
                return False
        elif model_name == 'sky':
            if model_name in self.include_list and self.include_sky:
                print(f"Sky Cubemap is visible, checked")
                return True
            else:
                return False
        elif model_name.startswith('obj_'):
            if model_name in self.include_list and self.include_obj:
                return True
            else:
                return False
        else:
            raise ValueError(f'Unknown model name {model_name}')

    def update_optimizer(self, iteration, stage):
        for model_name in self.model_name_id.keys():
            model: GaussianModel = getattr(self, model_name)
            if model_name == 'background':
                model.update_optimizer()
            else:
                model.update_optimizer(iteration=iteration, stage=stage)

        if self.actor_pose is not None:
            self.actor_pose.update_optimizer()

        if self.color_correction is not None:
            self.color_correction.update_optimizer()
        if self.pose_correction is not None:
            self.pose_correction.update_optimizer()

    def get_box_reg_loss(self):
        box_reg_loss = 0.
        for obj_name in self.obj_list:
            obj_model: GaussianModelActor = getattr(self, obj_name)
            box_reg_loss += obj_model.box_reg_loss()
        box_reg_loss /= len(self.obj_list)

        return box_reg_loss


    def _init_scene_graph(self):
        self.scene_graph_nodes = {}
        self.scene_graph_root = SceneGraphNode("root", node_type="root", parent=None)
        self.scene_graph_nodes["root"] = self.scene_graph_root
        if self.include_background:
            self._register_scene_node("background", node_type="background", parent=self.scene_graph_root)
        for obj_name in self.obj_list:
            self._register_scene_node(obj_name, node_type="object", parent=self.scene_graph_root)

    def _register_scene_node(self, name, node_type="object", parent=None):
        if name in self.scene_graph_nodes:
            return self.scene_graph_nodes[name]
        if parent is None:
            parent = self.scene_graph_root
        node = SceneGraphNode(name=name, node_type=node_type, parent=parent)
        parent.children.append(node)
        self.scene_graph_nodes[name] = node
        return node

    def _update_scene_graph_node(self, name, center=None, extent=None, visible=None):
        if name not in self.scene_graph_nodes:
            parent = self.scene_graph_root
            node_type = "background" if name == "background" else "object"
            node = self._register_scene_node(name, node_type=node_type, parent=parent)
        else:
            node = self.scene_graph_nodes[name]
        node.update(center=center, extent=extent, visible=visible)
        return node

    def _frustum_cull_sphere(self, center, extent, viewpoint_camera, margin=0.05):
        """
        :,.
         True().
        """
        try:
            if center is None:
                return True
            # extent  tensor  [1], float
            radius = float(extent.max().item()) if isinstance(extent, torch.Tensor) else float(extent)
            device = center.device

            proj = viewpoint_camera.full_proj_transform
            if proj.device != device:
                proj = proj.to(device)

            center_h = torch.cat([center, torch.ones(1, device=device)], dim=0)
            clip = proj @ center_h
            w = clip[3]
            if torch.abs(w) < 1e-6:
                return True
            ndc = clip[:3] / w
            x, y, z = ndc

            dist = torch.clamp(torch.norm(center - viewpoint_camera.camera_center.to(device)), min=1.0)
            pad = float(margin) + float(radius) / float(dist.item())
            in_view = (
                (x >= (-1.0 - pad)) & (x <= (1.0 + pad)) &
                (y >= (-1.0 - pad)) & (y <= (1.0 + pad)) &
                (z >= (-1.0)) & (z <= (1.0 + pad))
            )
            return bool(in_view.item())
        except Exception:

            return True

def rotate_xyz(angle_x=0, angle_y=0, angle_z=0):
    ax = torch.deg2rad(torch.tensor(angle_x, device='cuda', dtype=torch.float32))
    ay = torch.deg2rad(torch.tensor(angle_y, device='cuda', dtype=torch.float32))
    az = torch.deg2rad(torch.tensor(angle_z, device='cuda', dtype=torch.float32))
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
    return R.T
