import torch
import torch.nn as nn
import numpy as np
import os
from scene.gaussian_model import GaussianModel
from utils.general_utils import quaternion_to_matrix, inverse_sigmoid, matrix_to_quaternion, get_expon_lr_func, quaternion_raw_multiply
from utils.sh_utils import RGB2SH, IDFT
from scene.dataset_readers import fetchPly
from plyfile import PlyData, PlyElement
from simple_knn._C import distCUDA2
import math
import time
from scene.embedding import Embedding
from einops import repeat
from torch_scatter import scatter_max
from utils.general_utils import quaternion_to_matrix
from scene.deform_model import DeformModel

class GaussianModelActor(GaussianModel):
    def __init__(
        self,
        model_name,
        obj_meta,
        args,
        sh_degree, feat_dim: int=32,
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
        # parse obj_meta
        super().__init__(sh_degree=sh_degree, args=args, model_name=model_name)

        # ===== Hash encoding: /(waymo_default.py  *_obj )=====

        if hasattr(args, "hash_levels_obj"):
            self.hash_levels = getattr(args, "hash_levels_obj")
        if hasattr(args, "hash_base_resolution_obj"):
            self.hash_base_resolution = getattr(args, "hash_base_resolution_obj")
        if hasattr(args, "hash_finest_resolution_obj"):
            self.hash_finest_resolution = getattr(args, "hash_finest_resolution_obj")
        if hasattr(args, "hash_log2_size_obj"):
            self.hash_log2_size = getattr(args, "hash_log2_size_obj")
        if hasattr(args, "use_hash_feat_single_level_obj"):
            self.use_hash_feat_single_level = bool(getattr(args, "use_hash_feat_single_level_obj"))
        if hasattr(args, "use_hash_feat_multi_level_obj"):
            self.use_hash_feat_multi_level = bool(getattr(args, "use_hash_feat_multi_level_obj"))

        if hasattr(args, "use_hash_interpolation"):
            self.use_hash_interpolation = bool(getattr(args, "use_hash_interpolation"))

        self.hash_encoding = None
        self.obj_meta = obj_meta
        self.obj_class = obj_meta['class']
        self.obj_class_label = obj_meta['class_label']
        self.deformable = obj_meta['deformable']
        self.start_frame = obj_meta['start_frame']
        self.start_timestamp = obj_meta['start_timestamp']
        self.end_frame = obj_meta['end_frame']
        self.end_timestamp = obj_meta['end_timestamp']
        self.track_id = obj_meta['track_id']
        self.optimizer = None
        # fourier spherical harmonics
        self.fourier_dim = args.fourier_dim
        self.fourier_scale = args.fourier_scale

        # bbox
        length, width, height = obj_meta['length'], obj_meta['width'], obj_meta['height']
        self.bbox = np.array([length, width, height]).astype(np.float32)
        xyz = torch.tensor(self.bbox).float().cuda()
        self.min_xyz, self.max_xyz =  -xyz/2., xyz/2.

        extent = max(length*1.5/args.box_scale, width*1.5/args.box_scale, height) / 2.
        self.extent = torch.tensor([extent]).float().cuda()

        num_classes = 1 if args.use_semantic else 0
        self.num_classes_global = args.num_classes if args.use_semantic else 0

        self.flip_prob = args.flip_prob if not self.deformable else 0.
        self.flip_axis = 1

        self.spatial_lr_scale = extent

        self.embedding_appearance = None

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
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist
        self.add_level = add_level
        self.progressive = progressive
        self.extend = extend
        self.visible_threshold = visible_threshold
        print(f"[GaussianModelActor] visible_threshold: {self.visible_threshold}")
        self.dist2level = dist2level
        self.base_layer = base_layer
        self.visible_mask = None
        self.rendered_anchor_mask = None
        self.sub_pos_offsets = torch.tensor([[i % fork, (i // fork) % fork, i // (fork * fork)] for i in range(fork**3)]).float().cuda()
        self._offset = torch.empty(0)
        self.offset_gradient_accum = torch.empty(0)
        self.offset_denom = torch.empty(0)
        self.obj_dist = torch.empty(0)
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
                nn.Sigmoid()   # nn.Sigmoid()
            ).cuda()

        max_lod_level = getattr(args, 'max_hash_level', None)
        if max_lod_level is None:
            if getattr(args, 'use_hash_encoding', False):
                max_lod_level = getattr(args, 'hash_levels', 12)
            else:
                max_lod_level = getattr(args, 'max_lod_level', 32)

        self.use_lod_bias = getattr(args, 'use_lod_bias', False)

        use_anchor_deform = getattr(args, 'use_anchor_deform', True)
        anchor_deform_feat_dim = getattr(args, 'anchor_deform_feat_dim', 32)
        anchor_deform_hidden = getattr(args, 'anchor_deform_hidden', 256)
        anchor_deform_layers = getattr(args, 'anchor_deform_layers', 8)
        anchor_deform_use_rotation = getattr(args, 'anchor_deform_use_rotation', True)
        anchor_deform_xyz_multires = getattr(args, 'anchor_deform_xyz_multires', 10)
        anchor_deform_t_multires = getattr(args, 'anchor_deform_t_multires', 10)

        if use_anchor_deform:
            print("  Anchor-Conditioned Deformation enabled")
            print(f"  - Network: {anchor_deform_layers} x {anchor_deform_hidden}")
            print(f"  - XYZ multires: {anchor_deform_xyz_multires}")
            print(f"  - Time multires: {anchor_deform_t_multires}")
            print(f"  - Rotation output: {anchor_deform_use_rotation}")

        self.deform = DeformModel(
            is_blender=False,
            is_6dof=False,
            max_lod_level=max_lod_level,
            use_anchor_deform=use_anchor_deform,
            anchor_deform_feat_dim=anchor_deform_feat_dim,
            anchor_deform_hidden=anchor_deform_hidden,
            anchor_deform_layers=anchor_deform_layers,
            anchor_deform_use_rotation=anchor_deform_use_rotation,
            anchor_deform_xyz_multires=anchor_deform_xyz_multires,
            anchor_deform_t_multires=anchor_deform_t_multires,
        )

    def _disable_actor_lod(self):
        flag = os.environ.get("DRIVESPLAT_DISABLE_ACTOR_LOD", "").lower()
        return flag in ("1", "true", "yes", "on")


    @property
    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        return self._anchor

    # @property

    def get_extent(self):
        max_scaling = torch.max(self.get_scaling, dim=1).values

        extent_lower_bound = torch.topk(max_scaling, int(self.get_anchor.shape[0] * 0.1), largest=False).values[-1] / self.percent_dense
        extent_upper_bound = torch.topk(max_scaling, int(self.get_anchor.shape[0] * 0.1), largest=True).values[-1] / self.percent_dense

        extent = torch.clamp(self.extent, min=extent_lower_bound, max=extent_upper_bound)
        print(f'extent: {extent.item()}, extent bound: [{extent_lower_bound}, {extent_upper_bound}]')

        return extent
    @property
    def get_semantic(self):
        semantic = torch.zeros((self.get_anchor.shape[0], self.num_classes_global)).float().cuda()
        if self.semantic_mode == 'logits':
            semantic[:, self.obj_class_label] = self._semantic[:, 0] # ubounded semantic
        elif self.semantic_mode == 'probabilities':
            semantic[:, self.obj_class_label] = torch.nn.functional.sigmoid(self[:, 0])

        return semantic

    def get_features_fourier(self, frame=0):
        normalized_frame = (frame - self.start_frame) / (self.end_frame - self.start_frame)
        time = self.fourier_scale * normalized_frame

        idft_base = IDFT(time, self.fourier_dim)[0].cuda()
        features_dc = self._features_dc
        features_dc = torch.sum(features_dc * idft_base[..., None], dim=1, keepdim=True)
        features_rest = self._features_rest # [N, sh, 3]
        features = torch.cat([features_dc, features_rest], dim=1) # [N, (sh + 1) * C, 3]
        return features

    def create_from_pcd(self, spatial_lr_scale, logger=None, model_path=None):
        pointcloud_path = os.path.join(model_path, 'input_ply', f'points3D_{self.model_name}.ply')
        if os.path.exists(pointcloud_path):
            pcd = fetchPly(pointcloud_path)
            pointcloud_xyz = np.asarray(pcd.points)
            if pointcloud_xyz.shape[0] < 2000:
                self.random_initialization = True
            else:
                self.random_initialization = False
        else:
            self.random_initialization = True

        if self.random_initialization is True:
            points_dim = 15   # pointcloud size = points_dim^3
            points_x, points_y, points_z = np.meshgrid(
                np.linspace(-1., 1., points_dim), np.linspace(-1., 1., points_dim), np.linspace(-1., 1., points_dim),
            )

            points_x = points_x.reshape(-1)
            points_y = points_y.reshape(-1)
            points_z = points_z.reshape(-1)

            bbox_xyz_scale = self.bbox / 2.
            pointcloud_xyz = np.stack([points_x, points_y, points_z], axis=-1)
            pointcloud_xyz = pointcloud_xyz * bbox_xyz_scale
            pointcloud_rgb = np.random.rand(*pointcloud_xyz.shape).astype(np.float32)
            print(f'Creating random pointcloud for {self.model_name}, pointcloud size: {pointcloud_xyz.shape[0]}')
        elif not self.deformable and self.flip_prob > 0.:
            pcd = fetchPly(pointcloud_path)
            pointcloud_xyz = np.asarray(pcd.points)
            pointcloud_rgb = np.asarray(pcd.colors)
            num_pointcloud_1 = (pointcloud_xyz[:, self.flip_axis] > 0).sum()
            num_pointcloud_2 = (pointcloud_xyz[:, self.flip_axis] < 0).sum()
            if num_pointcloud_1 >= num_pointcloud_2:
                pointcloud_xyz_part = pointcloud_xyz[pointcloud_xyz[:, self.flip_axis] > 0]
                pointcloud_rgb_part = pointcloud_rgb[pointcloud_xyz[:, self.flip_axis] > 0]
            else:
                pointcloud_xyz_part = pointcloud_xyz[pointcloud_xyz[:, self.flip_axis] < 0]
                pointcloud_rgb_part = pointcloud_rgb[pointcloud_xyz[:, self.flip_axis] < 0]
            pointcloud_xyz_flip = pointcloud_xyz_part.copy()
            pointcloud_xyz_flip[:, self.flip_axis] *= -1
            pointcloud_rgb_flip = pointcloud_rgb_part.copy()
            pointcloud_xyz = np.concatenate([pointcloud_xyz, pointcloud_xyz_flip], axis=0)
            pointcloud_rgb = np.concatenate([pointcloud_rgb, pointcloud_rgb_flip], axis=0)
        else:
            pcd = fetchPly(pointcloud_path)
            pointcloud_xyz = np.asarray(pcd.points)
            pointcloud_rgb = np.asarray(pcd.colors)


        points = torch.tensor(pointcloud_xyz).float().cuda()
        self.spatial_lr_scale = self.spatial_lr_scale
        try:
            #  hash  per-axis  AABB(xyz_min/xyz_max),.
            box_min = torch.min(points) * self.extend
            print(f"box_min: {box_min}")
            box_max = torch.max(points) * self.extend
            print(f"box_max: {box_max}")
            box_d = box_max - box_min
            print(f"box_d: {box_d}")
            xyz_min = torch.min(points, dim=0).values * self.extend
            xyz_max = torch.max(points, dim=0).values * self.extend
            self.obj_dist = box_d
            if self.base_layer < 0:
                default_voxel_size = 0.02
                self.base_layer = torch.round(torch.log2(box_d/default_voxel_size)).int().item()-(self.levels//2)+1
            print(f"base_layer: {self.base_layer}, fork: {self.fork}")
            self.voxel_size = box_d/(float(self.fork) ** self.base_layer)
            pre_voxel_size = self.voxel_size.clone() if isinstance(self.voxel_size, torch.Tensor) else self.voxel_size
            #  hash_sample
            try:
                self.pre_hash_base_voxel = pre_voxel_size.max().detach().cpu() if isinstance(pre_voxel_size, torch.Tensor) else float(pre_voxel_size)
            except Exception:
                self.pre_hash_base_voxel = float(pre_voxel_size) if not isinstance(pre_voxel_size, torch.Tensor) else float(pre_voxel_size.max().item())
            self.init_pos = torch.tensor([box_min, box_min, box_min]).float().cuda()
            used_hash = bool(getattr(self, 'use_hash_encoding', False))
            print(f"voxel_size: {self.voxel_size}, init_pos: {self.init_pos}")
            print(f"[{self.model_name}] init sampling: use_hash_encoding={used_hash}")
            if used_hash:
                self.hash_sample(points, xyz_min, xyz_max)
                # hash  voxel_size  level0  hash ( offset/LOD )
                if getattr(self, "hash_encoding", None) is not None:
                    res0 = int(self.hash_encoding.level_resolution(0))
                    base_voxel_vec = (xyz_max - xyz_min) / float(res0)
                    self.voxel_size = torch.max(base_voxel_vec).float().to(device="cuda")
            else:
                self.lod_tree_sample(points, self.init_pos)
            print(f"visible_threshold: {self.visible_threshold}")
            if self.visible_threshold < 0:
                self.visible_threshold = 0.0
                self.positions, self._level, self.visible_threshold, _ = self.weed_out(self.positions, self._level)
                print(f"visible_threshold: {self.visible_threshold}")
            else:
                self.positions, self._level, _, _ = self.weed_out(self.positions, self._level)
        except Exception as e:
            print(f"points_min: {torch.min(points)}, points_max: {torch.max(points)}, extend: {self.extend}")
            print(f"Error at initialisation: {e}")

        logger.info(f'Branches of Tree: {self.fork}')
        logger.info(f'Use Hash Encoding: {used_hash}')
        if used_hash:
            logger.info(f'Pre-hash Base Layer: {self.base_layer} (computed from box)')
            logger.info(f'Pre-hash Voxel Size: {pre_voxel_size}')
            logger.info(f'Hash Params: hash_levels={getattr(self, "hash_levels", None)}, hash_base_resolution={getattr(self, "hash_base_resolution", None)}, hash_finest_resolution={getattr(self, "hash_finest_resolution", None)}, hash_log2_size={getattr(self, "hash_log2_size", None)}')
            logger.info(f'Actor LOD Levels (final): {self.levels}')
            logger.info(f'Final Voxel Size (post-hash): {self.voxel_size}')
        else:
            logger.info(f'Base Layer of Tree: {self.base_layer}')
            logger.info(f'Actor LOD Levels: {self.levels}')
            logger.info(f'Max Voxel Size: {self.voxel_size}')
            logger.info(f'Min Voxel Size: {self.voxel_size/(2.0 ** (self.levels - 1))}')


        offsets = torch.zeros((self.positions.shape[0], self.n_offsets, 3)).float().cuda()
        anchors_feat = torch.zeros((self.positions.shape[0], self.feat_dim)).float().cuda()
        print(f"Number of points at initialisation for {self.model_name}: ", points.shape[0])
        dist2 = torch.clamp_min(distCUDA2(self.positions).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        rots = torch.zeros((self.positions.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((self.positions.shape[0], 1), dtype=torch.float, device="cuda"))
        semamtics = torch.zeros((self.positions.shape[0], self.num_classes), dtype=torch.float, device="cuda")

        # ,positions
        if self.random_initialization:

            fused_color = torch.rand(self.positions.shape[0], 3).float().cuda()
        else:


            fused_color = torch.zeros(self.positions.shape[0], 3).float().cuda()


        features = torch.zeros((self.positions.shape[0], 3, (self.max_sh_degree + 1) ** 2)).float().cuda()
        features[:, :3, 0] = fused_color
        features[:, 3:, 1:] = 0.0


        self._features_dc = nn.Parameter(features[:,:,0:1].transpose(1, 2).contiguous().requires_grad_(True))
        self._features_rest = nn.Parameter(features[:,:,1:].transpose(1, 2).contiguous().requires_grad_(True))


        self._anchor = nn.Parameter(self.positions.requires_grad_(True))
        self._offset = nn.Parameter(offsets.requires_grad_(True))
        self._anchor_feat = nn.Parameter(anchors_feat.requires_grad_(True))
        self._scaling = nn.Parameter(scales.requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._opacity = nn.Parameter(opacities.requires_grad_(False))
        self._semantic = nn.Parameter(semamtics.requires_grad_(True))
        self._level = self._level.unsqueeze(dim=1)
        self._extra_level = torch.zeros(self._anchor.shape[0], dtype=torch.float, device="cuda")
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")


    def training_setup(self, training_args):
        tag = 'obj'
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self._anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self._anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self._anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self._anchor.shape[0], 1), device="cuda")

        self.deform.train_setting(training_args)


        l = [
            {'params': [self._anchor], 'lr': training_args.position_lr_init * self.spatial_lr_scale, "name": "anchor"},
            {'params': [self._offset], 'lr': training_args.offset_lr_init * self.spatial_lr_scale, "name": "offset"},
            {'params': [self._anchor_feat], 'lr': training_args.feature_lr, "name": "anchor_feat"},
            {'params': [self._opacity], 'lr': training_args.opacity_lr, "name": "opacity"},
            {'params': [self._scaling], 'lr': training_args.scaling_lr, "name": "scaling"},
            {'params': [self._rotation], 'lr': training_args.rotation_lr, "name": "rotation"},
            {'params': [self._semantic], 'lr': training_args.semantic_lr, "name": "semantic"},
            {'params': self.mlp_opacity.parameters(), 'lr': training_args.mlp_opacity_lr_init, "name": "mlp_opacity"},
            {'params': self.mlp_cov.parameters(), 'lr': training_args.mlp_cov_lr_init, "name": "mlp_cov"},
            {'params': self.mlp_color.parameters(), 'lr': training_args.mlp_color_lr_init, "name": "mlp_color"},
        ]

        # appearance
        if self.appearance_dim > 0:
            l.append({'params': self.embedding_appearance.parameters(),
                     'lr': training_args.appearance_lr_init,
                     "name": "embedding_appearance"})

        # feature bank
        if self.use_feat_bank:
            l.append({'params': self.mlp_feature_bank.parameters(),
                     'lr': training_args.mlp_featurebank_lr_init,
                     "name": "mlp_featurebank"})

        if self.use_hash_encoding and hasattr(self, 'hash_encoding') and self.hash_encoding is not None:
            grid_lr = getattr(training_args, 'grid_lr_init', 0.01)  # spatial_lr_scale!
            l.append({'params': self.hash_encoding.parameters(), 'lr': grid_lr, "name": "hash_grid"})
            print(f"[{self.model_name}]  Added hash_encoding to optimizer (lr={grid_lr:.6f})")
            if hasattr(self, 'hash_feat_multi_proj') and self.hash_feat_multi_proj is not None:
                l.append({'params': self.hash_feat_multi_proj.parameters(), 'lr': training_args.feature_lr, "name": "hash_feat_multi_proj"})
                print(f"[{self.model_name}]  Added hash_feat_multi_proj to optimizer (lr={training_args.feature_lr})")

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)


        self.anchor_scheduler_args = get_expon_lr_func(
            lr_init=training_args.position_lr_init*self.spatial_lr_scale,
            lr_final=training_args.position_lr_final*self.spatial_lr_scale,
            lr_delay_mult=training_args.position_lr_delay_mult,
            max_steps=training_args.position_lr_max_steps)

        self.offset_scheduler_args = get_expon_lr_func(
            lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
            lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
            lr_delay_mult=training_args.offset_lr_delay_mult,
            max_steps=training_args.offset_lr_max_steps)

        hash_grid_max_steps = getattr(training_args, 'iterations', 60000)
        self.hash_grid_scheduler_args = get_expon_lr_func(
            lr_init=training_args.grid_lr_init,
            lr_final=training_args.grid_lr_final,
            lr_delay_mult=0.1,
            max_steps=hash_grid_max_steps)

        self.mlp_opacity_scheduler_args = get_expon_lr_func(
            lr_init=training_args.mlp_opacity_lr_init,
            lr_final=training_args.mlp_opacity_lr_final,
            lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
            max_steps=training_args.mlp_opacity_lr_max_steps)

        self.mlp_cov_scheduler_args = get_expon_lr_func(
            lr_init=training_args.mlp_cov_lr_init,
            lr_final=training_args.mlp_cov_lr_final,
            lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
            max_steps=training_args.mlp_cov_lr_max_steps)

        self.mlp_color_scheduler_args = get_expon_lr_func(
            lr_init=training_args.mlp_color_lr_init,
            lr_final=training_args.mlp_color_lr_final,
            lr_delay_mult=training_args.mlp_color_lr_delay_mult,
            max_steps=training_args.mlp_color_lr_max_steps)

        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(
                lr_init=training_args.mlp_featurebank_lr_init,
                lr_final=training_args.mlp_featurebank_lr_final,
                lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                max_steps=training_args.mlp_featurebank_lr_max_steps)

        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(
                lr_init=training_args.appearance_lr_init,
                lr_final=training_args.appearance_lr_final,
                lr_delay_mult=training_args.appearance_lr_delay_mult,
                max_steps=training_args.appearance_lr_max_steps)

        self.denom = torch.zeros((self._anchor.shape[0], 1), device="cuda")
        self.active_sh_degree = 0

    def set_max_radii(self, visibility_obj, max_radii2D):
        self.max_radii2D[visibility_obj] = torch.max(self.max_radii2D[visibility_obj], max_radii2D[visibility_obj])

    def box_reg_loss(self):
        scaling_max = self.get_scaling.max(dim=1).values
        scaling_max = torch.where(scaling_max > self.extent * self.percent_dense, scaling_max, 0.)
        reg_loss = (scaling_max / self.extent).mean()

        return reg_loss

    def set_level(self, points, cameras, scales, dist_ratio=0.95, init_level=-1, levels=-1, model_path=None):
        pointcloud_path = os.path.join(model_path, 'input_ply', f'points3D_{self.model_name}.ply')
        if os.path.exists(pointcloud_path):
            pcd = fetchPly(pointcloud_path)
            pointcloud_xyz = np.asarray(pcd.points)
            if pointcloud_xyz.shape[0] < 2000:
                self.random_initialization = True
            else:
                self.random_initialization = False
        else:
            self.random_initialization = True
        if self.random_initialization is True:
            points_dim = 15   # pointcloud size = points_dim^3
            points_x, points_y, points_z = np.meshgrid(
                np.linspace(-1., 1., points_dim), np.linspace(-1., 1., points_dim), np.linspace(-1., 1., points_dim),
            )
            points_x = points_x.reshape(-1)
            points_y = points_y.reshape(-1)
            points_z = points_z.reshape(-1)
            bbox_xyz_scale = self.bbox / 2.
            pointcloud_xyz = np.stack([points_x, points_y, points_z], axis=-1)
            pointcloud_xyz = pointcloud_xyz * bbox_xyz_scale
            pointcloud_rgb = np.random.rand(*pointcloud_xyz.shape).astype(np.float32)
            print(f'Creating random pointcloud for {self.model_name}, pointcloud size: {pointcloud_xyz.shape[0]}')
        elif not self.deformable and self.flip_prob > 0.:
            pcd = fetchPly(pointcloud_path)
            pointcloud_xyz = np.asarray(pcd.points)
            pointcloud_rgb = np.asarray(pcd.colors)
            num_pointcloud_1 = (pointcloud_xyz[:, self.flip_axis] > 0).sum()
            num_pointcloud_2 = (pointcloud_xyz[:, self.flip_axis] < 0).sum()
            if num_pointcloud_1 >= num_pointcloud_2:
                pointcloud_xyz_part = pointcloud_xyz[pointcloud_xyz[:, self.flip_axis] > 0]
                pointcloud_rgb_part = pointcloud_rgb[pointcloud_xyz[:, self.flip_axis] > 0]
            else:
                pointcloud_xyz_part = pointcloud_xyz[pointcloud_xyz[:, self.flip_axis] < 0]
                pointcloud_rgb_part = pointcloud_rgb[pointcloud_xyz[:, self.flip_axis] < 0]
            pointcloud_xyz_flip = pointcloud_xyz_part.copy()
            pointcloud_xyz_flip[:, self.flip_axis] *= -1
            pointcloud_rgb_flip = pointcloud_rgb_part.copy()
            pointcloud_xyz = np.concatenate([pointcloud_xyz, pointcloud_xyz_flip], axis=0)
            pointcloud_rgb = np.concatenate([pointcloud_rgb, pointcloud_rgb_flip], axis=0)
        else:
            pcd = fetchPly(pointcloud_path)
            pointcloud_xyz = np.asarray(pcd.points)
            pointcloud_rgb = np.asarray(pcd.colors)


        points = torch.tensor(pointcloud_xyz).float().cuda()
        self.spatial_lr_scale = self.spatial_lr_scale
        box_min = torch.min(points)*self.extend
        print(f"box_min: {box_min}")
        box_max = torch.max(points)*self.extend
        print(f"box_max: {box_max}")
        box_d = box_max - box_min
        print(f"box_d: {box_d}")
        self.obj_dist = box_d


        all_dist = torch.tensor([]).cuda()
        self.cam_infos = torch.empty(0, 4).float().cuda()
        for scale in scales:
            for cam in cameras[scale]:
                cam_center = cam.camera_center
                cam_info = torch.tensor([cam_center[0], cam_center[1], cam_center[2], scale]).float().cuda()
                self.cam_infos = torch.cat((self.cam_infos, cam_info.unsqueeze(dim=0)), dim=0)
                dist = torch.sqrt(torch.sum((points - cam_center)**2, dim=1))
                dist_max = torch.quantile(dist, dist_ratio)
                dist_min = torch.quantile(dist, 1 - dist_ratio)
                new_dist = torch.tensor([dist_min, dist_max]).float().cuda()
                new_dist = new_dist * scale
                all_dist = torch.cat((all_dist, new_dist), dim=0)
        dist_max = torch.quantile(all_dist, dist_ratio)
        dist_min = torch.quantile(all_dist, 1 - dist_ratio)
        self.base_standard_dist = dist_max
        self.standard_dist = dist_max
        self.base_dist_ratio = dist_ratio
        self.current_dist_ratio = dist_ratio

        if levels == -1:
            # hash  hash_levels(,/)
            if getattr(self, "use_hash_encoding", False) and getattr(self, "hash_levels", None) is not None and int(getattr(self, "hash_levels")) > 0:
                self.levels = int(self.hash_levels)
            else:
                self.levels = torch.round(torch.log2(self.obj_dist)/math.log2(self.fork)).int().item() + 1
        else:
            self.levels = levels
        if init_level == -1:
            self.init_level = int(self.levels/2)
        else:
            self.init_level = init_level

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
        print(f"[{self.model_name}] Building LOD tree time: {int(time_diff // 60)} min {time_diff % 60} sec")

    def plot_levels(self):
        for level in range(self.levels):
            level_mask = (self._level == level).squeeze(dim=1)
            print(f'Level {level}: {torch.sum(level_mask).item()}, Ratio: {torch.sum(level_mask).item()/self._level.shape[0]}')

    def save_deform_weights(self, path, iteration):
        model_name = self.model_name
        self.deform.save_weights(path, iteration, model_name)

    def load_deform_weights(self, path, iteration):
        model_name = self.model_name
        self.deform.load_weights(path, iteration, model_name)
        print(f"Deformation weights loaded from {path} for iteration {iteration} and model {model_name}")

    def load_ply_sparse_gaussian(self, path=None, input_ply=None):
        if path is None:
            plydata = input_ply
        else:
            plydata = PlyData.read(path)
            plydata = plydata.elements[0]


        try:
            self.voxel_size = torch.tensor(plydata["info"][0]).float()
            self.standard_dist = torch.tensor(plydata["info"][1]).float()
        except (ValueError, KeyError):
            #  info ,
            print(f"Warning: 'info' field not found in PLY file. Using default values.")
            self.voxel_size = torch.tensor(0.02).float().cuda()
            self.standard_dist = torch.tensor(1.0).float().cuda()

        anchor = np.stack((np.asarray(plydata["x"]),
                        np.asarray(plydata["y"]),
                        np.asarray(plydata["z"])),  axis=1).astype(np.float32)
        levels = np.asarray(plydata["level"])[... ,np.newaxis].astype(np.int)
        extra_levels = np.asarray(plydata["extra_level"])[... ,np.newaxis].astype(np.float32)

        opacities = np.asarray(plydata["opacity"])[..., np.newaxis].astype(np.float32)

        scale_names = [p.name for p in plydata.properties if p.name.startswith("scale_")]
        scale_names = sorted(scale_names, key = lambda x: int(x.split('_')[-1]))
        scales = np.zeros((anchor.shape[0], len(scale_names)))
        for idx, attr_name in enumerate(scale_names):
            scales[:, idx] = np.asarray(plydata[attr_name]).astype(np.float32)

        rot_names = [p.name for p in plydata.properties if p.name.startswith("rot")]
        rot_names = sorted(rot_names, key = lambda x: int(x.split('_')[-1]))
        rots = np.zeros((anchor.shape[0], len(rot_names)))
        for idx, attr_name in enumerate(rot_names):
            rots[:, idx] = np.asarray(plydata[attr_name]).astype(np.float32)

        # anchor_feat
        anchor_feat_names = [p.name for p in plydata.properties if p.name.startswith("f_anchor_feat")]
        anchor_feat_names = sorted(anchor_feat_names, key = lambda x: int(x.split('_')[-1]))
        anchor_feats = np.zeros((anchor.shape[0], len(anchor_feat_names)))
        for idx, attr_name in enumerate(anchor_feat_names):
            anchor_feats[:, idx] = np.asarray(plydata[attr_name]).astype(np.float32)

        offset_names = [p.name for p in plydata.properties if p.name.startswith("f_offset")]
        offset_names = sorted(offset_names, key = lambda x: int(x.split('_')[-1]))
        offsets = np.zeros((anchor.shape[0], len(offset_names)))
        for idx, attr_name in enumerate(offset_names):
            offsets[:, idx] = np.asarray(plydata[attr_name]).astype(np.float32)
        offsets = offsets.reshape((offsets.shape[0], 3, -1))

        semantic_names = [p.name for p in plydata.properties if p.name.startswith("semantic")]
        semantic_names = sorted(semantic_names, key = lambda x: int(x.split('_')[-1]))
        semantic = np.zeros((anchor.shape[0], len(semantic_names)))
        for idx, attr_name in enumerate(semantic_names):
            semantic[:, idx] = np.asarray(plydata[attr_name]).astype(np.float32)

        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self._level = torch.tensor(levels, dtype=torch.int, device="cuda")
        self._extra_level = torch.tensor(extra_levels, dtype=torch.float, device="cuda").squeeze(dim=1)
        self._offset = nn.Parameter(torch.tensor(offsets, dtype=torch.float, device="cuda").transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(torch.tensor(anchor, dtype=torch.float, device="cuda").requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._semantic = nn.Parameter(torch.tensor(semantic, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(torch.tensor(rots, dtype=torch.float, device="cuda").requires_grad_(False))
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")
        self.levels = torch.max(self._level) - torch.min(self._level) + 1
        self.active_sh_degree = self.max_sh_degree


    def weed_out(self, anchor_positions, anchor_levels):
        if self._disable_actor_lod():
            keep_mask = torch.ones(anchor_positions.shape[0], dtype=torch.bool, device=anchor_positions.device)
            mean_visible = torch.ones((), dtype=torch.float32, device=anchor_positions.device)
            return anchor_positions, anchor_levels, mean_visible, keep_mask

        visible_count = torch.zeros(anchor_positions.shape[0], dtype=torch.int, device="cuda")
        if self.use_hash_encoding and getattr(self, "hash_encoding", None) is not None:
            lod_base = float(self.hash_encoding.per_level_scale)
        else:
            lod_base = float(self.fork)
        for cam in self.cam_infos:
            cam_center, scale = cam[:3], cam[3]
            dist = torch.sqrt(torch.sum((anchor_positions - cam_center)**2, dim=1)) * scale
            pred_level = torch.log(self.standard_dist/dist) / math.log(lod_base)
            int_level = self.map_to_int_level(pred_level, self.levels - 1)
            visible_count += (anchor_levels <= int_level).int()
        visible_count = visible_count/len(self.cam_infos)
        weed_mask = (visible_count > self.visible_threshold)
        mean_visible = torch.mean(visible_count)
        return anchor_positions[weed_mask], anchor_levels[weed_mask], mean_visible, weed_mask

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()


    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "hash_grid":
                lr = self.hash_grid_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_opacity":
                lr = self.mlp_opacity_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_cov":
                lr = self.mlp_cov_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "mlp_color":
                lr = self.mlp_color_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.use_feat_bank and param_group["name"] == "mlp_featurebank":
                lr = self.mlp_featurebank_scheduler_args(iteration)
                param_group['lr'] = lr
            if self.appearance_dim > 0 and param_group["name"] == "embedding_appearance":
                lr = self.appearance_scheduler_args(iteration)
                param_group['lr'] = lr

    def update_optimizer(self, iteration, stage):
        #  iteration ,
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        if self.obj_class == 'pedestrian' and stage == 'fine':
            if hasattr(self.deform, 'optimizer') and self.deform.optimizer is not None:
                self.deform.optimizer.step()
                self.deform.optimizer.zero_grad()
            self.deform.update_learning_rate(iteration)

    def training_statis(self, viewspace_point_tensor_grad, opacity, visibility_mask, opacity_mask, anchor_mask):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity < 0] = 0

        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_mask] += temp_opacity.sum(dim=1, keepdim=True)

        # update anchor visiting statis
        self.anchor_demon[anchor_mask] += 1

        # update neural gaussian statis
        anchor_mask = anchor_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        # True
        # Count the number of True values in the anchor_mask
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_mask] = opacity_mask
        temp_mask = combined_mask.clone()
        if visibility_mask.size(0) > combined_mask[temp_mask].size(0):
            combined_mask[temp_mask] = visibility_mask[:combined_mask[temp_mask].shape[0]]
            nnn = visibility_mask[:combined_mask[temp_mask].shape[0]]
            viewspace_point_tensor_grad = viewspace_point_tensor_grad[:combined_mask[temp_mask].shape[0]]
            grad_norm = torch.norm(viewspace_point_tensor_grad[nnn, :2], dim=-1, keepdim=True)
            self.offset_gradient_accum[combined_mask] += grad_norm
            self.offset_denom[combined_mask] += 1
        elif visibility_mask.size(0) < combined_mask[temp_mask].size(0):
            combined_mask[temp_mask][:visibility_mask.size(0)] = visibility_mask
            viewspace_point_tensor_grad = viewspace_point_tensor_grad[:visibility_mask.shape[0]]
            grad_norm = torch.norm(viewspace_point_tensor_grad[visibility_mask, :2], dim=-1, keepdim=True)
            min_size = min(self.offset_gradient_accum[combined_mask].size(0), grad_norm.size(0))
            self.offset_gradient_accum[combined_mask][:min_size] += grad_norm[:min_size]
            self.offset_denom[combined_mask][:visibility_mask.size(0)] += 1
        else:
            combined_mask[temp_mask] = visibility_mask
            grad_norm = torch.norm(viewspace_point_tensor_grad[visibility_mask, :2], dim=-1, keepdim=True)

            self.offset_gradient_accum[combined_mask] += grad_norm
            self.offset_denom[combined_mask] += 1


    def set_anchor_mask(self, cam_center, iteration, resolution_scale):
        """
         :Anchor(Actor)

        :
        - U_d (Eq.11): hard selection
        - U_b (Eq.12): LOD bias _isoft selection
        """
        if self._disable_actor_lod():
            self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device=self._anchor.device)
            if hasattr(self, '_opacity_scale'):
                self._opacity_scale = torch.ones(self._anchor.shape[0], device=self._anchor.device)
            return

        # hash encodingLOD,anchor
        if self.use_hash_encoding and self.hash_disable_lod:
            self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device=self._anchor.device)
            if hasattr(self, '_opacity_scale'):
                self._opacity_scale = torch.ones(self._anchor.shape[0], device=self._anchor.device)
            return

        # ========== anchor ==========
        # hash  level  per_level_scale( fork=2)
        if self.use_hash_encoding and getattr(self, "hash_encoding", None) is not None:
            lod_base = float(self.hash_encoding.per_level_scale)
        else:
            lod_base = float(self.fork)
        lod_base_t = torch.tensor(lod_base, device=self._level.device, dtype=torch.float32)
        scale = torch.pow(lod_base_t, self._level.float())
        anchor_pos = self.get_anchor + (self.voxel_size / 2) / scale
        dist = torch.sqrt(torch.sum((anchor_pos - cam_center) ** 2, dim=1)) * resolution_scale
        dist_world = dist / resolution_scale if resolution_scale != 0 else dist

        # ========== LOD level ==========
        effective_standard_dist = self.standard_dist
        log_ratio = torch.log(effective_standard_dist / (dist + 1e-6)) / math.log(lod_base) + self._extra_level

        # bias
        depth_level_bias = torch.zeros_like(log_ratio)
        depth_level_bias = depth_level_bias + (dist_world > self.lod_depth_near).float() * self.lod_depth_bias_mid
        depth_level_bias = depth_level_bias + (dist_world > self.lod_depth_mid).float() * (self.lod_depth_bias_far - self.lod_depth_bias_mid)
        log_ratio_adjusted = log_ratio - depth_level_bias

        # ========== level(progressive training) ==========
        is_training = self.get_color_mlp.training
        if self.progressive and is_training:
            coarse_index = np.searchsorted(self.coarse_intervals, iteration) + 1 + self.init_level
        else:
            coarse_index = self.levels

        max_level = coarse_index - 1
        if self.use_hash_encoding and getattr(self, 'hash_levels', None) is not None and int(getattr(self, 'hash_levels', 0)) > 0:
            max_level = min(max_level, int(self.hash_levels) - 1)
        if iteration is not None and iteration < self.lod_warmup_iters:
            max_level = min(max_level, self.lod_warmup_max_level)

        # ==========  U_d (Eq.11): Hard threshold ==========
        target_level_hard = torch.floor(
            torch.clamp(log_ratio_adjusted, min=0, max=max_level)
        )
        mask_Ud = (self._level.squeeze(-1) <= target_level_hard)

        # ==========  U_b (Eq.12): Soft threshold with learnable bias ==========
        if self.use_lod_bias and self._lod_bias is not None:
            current_n_anchors = self._anchor.shape[0]

            # ,padding/truncate(forward)
            if self._lod_bias.shape[0] != current_n_anchors:
                if self._lod_bias.shape[0] < current_n_anchors:
                    # Anchors,0
                    padding = torch.zeros(current_n_anchors - self._lod_bias.shape[0],
                                         device=self._lod_bias.device, dtype=self._lod_bias.dtype)
                    lod_bias_padded = torch.cat([self._lod_bias.detach(), padding], dim=0)
                else:
                    # Anchors,
                    lod_bias_padded = self._lod_bias[:current_n_anchors].detach()
            else:
                lod_bias_padded = self._lod_bias.detach()
            target_level_soft = log_ratio_adjusted + lod_bias_padded
            mask_Ub = (self._level.squeeze(-1) <= target_level_soft)


            if iteration is not None and iteration < self.lod_warmup_iters_bias:
                alpha = iteration / self.lod_warmup_iters_bias
            else:
                alpha = 1.0

            if self.lod_adaptive_strategy == "Ud":
                self._anchor_mask = mask_Ud
            elif self.lod_adaptive_strategy == "Ub":
                self._anchor_mask = mask_Ub
            else:  # "mixed"
                self._anchor_mask = mask_Ud | (mask_Ub & (torch.rand_like(mask_Ub.float(), device=mask_Ub.device) < alpha))

            # ==========  Opacity (Eq.13) ==========
            # _opacity_scale,
            in_Ub_not_Ud = mask_Ub & (~mask_Ud)
            if not hasattr(self, '_opacity_scale') or self._opacity_scale is None or self._opacity_scale.shape[0] != current_n_anchors:
                self._opacity_scale = torch.ones(current_n_anchors, device=self._anchor.device)

            # opacity scale,_lod_bias

        else:
            # LOD bias,U_d
            self._anchor_mask = mask_Ud
            if hasattr(self, '_opacity_scale'):
                self._opacity_scale = torch.ones(self._anchor.shape[0], device=self._anchor.device)

    def set_anchor_mask_perlevel(self, cam_center, resolution_scale, cur_level):
        if self._disable_actor_lod():
            self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device=self._anchor.device)
            return

        if self.use_hash_encoding and getattr(self, "hash_encoding", None) is not None:
            lod_base = float(self.hash_encoding.per_level_scale)
        else:
            lod_base = float(self.fork)
        lod_base_t = torch.tensor(lod_base, device=self._level.device, dtype=torch.float32)
        scale = torch.pow(lod_base_t, self._level.float())
        anchor_pos = self._anchor + (self.voxel_size/2) / scale
        dist = torch.sqrt(torch.sum((anchor_pos - cam_center)**2, dim=1)) * resolution_scale
        effective_standard_dist = self.standard_dist
        pred_level = torch.log(effective_standard_dist/dist) / math.log(lod_base) + self._extra_level
        dist_world = dist / resolution_scale if resolution_scale != 0 else dist
        level_bias = torch.zeros_like(pred_level)
        level_bias = level_bias + (dist_world > self.lod_depth_near).float() * self.lod_depth_bias_mid
        level_bias = level_bias + (dist_world > self.lod_depth_mid).float() * (self.lod_depth_bias_far - self.lod_depth_bias_mid)
        pred_level = torch.clamp(pred_level - level_bias, min=0.0)
        int_level = self.map_to_int_level(pred_level, cur_level)
        self._anchor_mask = (self._level.squeeze(dim=1) <= int_level)


    def adjust_anchor(self, iteration, check_interval=100, success_threshold=0.8, grad_threshold=0.0002,
                      update_ratio=0.5, extra_ratio=4.0, extra_up=0.25, min_opacity=0.005, viewpoint_camera=None):
        # # adding anchors
        if self.offset_denom.shape[0] <= 0 :
            print(f"input offset_denom of class {self.obj_class} of {self.model_name} is already not right ")
        grads = self.offset_gradient_accum / self.offset_denom
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval * success_threshold * 0.5).squeeze(dim=1)
        self.anchor_growing(iteration, grads_norm, grad_threshold, update_ratio, extra_ratio, extra_up,
                             offset_mask, viewpoint_camera)
        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros(
            [self.get_anchor.shape[0] * self.n_offsets - self.offset_denom.shape[0], 1],
            dtype=torch.int32,
            device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros(
            [self.get_anchor.shape[0] * self.n_offsets - self.offset_gradient_accum.shape[0], 1],
            dtype=torch.int32,
            device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum],
                                                dim=0)

        # # prune anchors
        if self.get_anchor.shape[0] <= 0:
            print(f"current {self.obj_class} is not ok, {self.model_name}")

        if self.get_anchor.shape[0] > 0:
            if self.obj_class == 'pedestrian':
                min_opacity = 0.0005
            prune_mask = (self.opacity_accum < min_opacity * self.anchor_demon).squeeze(dim=1)
            anchors_mask = (self.anchor_demon > check_interval * success_threshold).squeeze(dim=1)
            prune_mask = torch.logical_and(prune_mask, anchors_mask)
            # Prune points outside the tracking box
            repeat_num = 2
            stds = self.get_scaling[:, :3]
            stds = stds[:, None, :].expand(-1, repeat_num, -1)
            means = torch.zeros_like(self._anchor)
            means = means[:, None, :].expand(-1, repeat_num, -1)
            samples = torch.normal(mean=means, std=stds)
            single_rot = getattr(self, 'obj_rot_single', self.obj_rots[0] if hasattr(self, 'obj_rots') and self.obj_rots is not None else torch.zeros(4, device='cuda'))
            single_trans = getattr(self, 'obj_trans_single', self.obj_trans[0] if hasattr(self, 'obj_trans') and self.obj_trans is not None else torch.zeros(3, device='cuda'))
            obj_rot = single_rot.expand(self.get_anchor.shape[0], -1)
            obj_trans = single_trans.unsqueeze(0).expand(self.get_anchor.shape[0], -1)
            rots = self.get_rotation
            rots = quaternion_raw_multiply(obj_rot, rots)
            rots = quaternion_to_matrix(rots)
            rots = rots[:, None, :, :].expand(-1, repeat_num, -1, -1)
            obj_rot = quaternion_to_matrix(obj_rot)
            temp = torch.einsum('bij, bj -> bi', obj_rot, self.get_anchor) + obj_trans
            origins = temp[:, None, :].expand(-1, repeat_num, -1)

            samples_xyz = torch.matmul(rots, samples.unsqueeze(-1)).squeeze(-1) + origins
            num_gaussians = self._anchor.shape[0]
            # min_xyz, max_xyz = -torch.abs(xyz / 2.), torch.abs(xyz / 2.)
            # min_xyz[0] = - torch.abs(min_xyz[0])
            # min_xyz[1] = - torch.abs(min_xyz[1])
            # max_xyz[0] = torch.abs(max_xyz[0])
            # max_xyz[1] = torch.abs(max_xyz[1])
            points_inside_box = torch.logical_and(
                torch.all((samples_xyz >= self.min_xyz).view(num_gaussians, -1), dim=-1),
                torch.all((samples_xyz <= self.max_xyz).view(num_gaussians, -1), dim=-1),
            )
            points_outside_box = torch.logical_not(points_inside_box)

            # update offset_denom
            offset_denom = self.offset_denom.view([-1, self.n_offsets])[~prune_mask]
            offset_denom = offset_denom.view([-1, 1])
            del self.offset_denom
            self.offset_denom = offset_denom

            offset_gradient_accum = self.offset_gradient_accum.view([-1, self.n_offsets])[~prune_mask]
            offset_gradient_accum = offset_gradient_accum.view([-1, 1])
            del self.offset_gradient_accum
            self.offset_gradient_accum = offset_gradient_accum

            # update opacity accum
            if anchors_mask.sum() > 0:
                self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
                self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()

            temp_opacity_accum = self.opacity_accum[~prune_mask]
            del self.opacity_accum
            self.opacity_accum = temp_opacity_accum

            temp_anchor_demon = self.anchor_demon[~prune_mask]
            del self.anchor_demon
            self.anchor_demon = temp_anchor_demon

            if prune_mask.shape[0] > 0:
                self.prune_anchor(prune_mask)
                if self._anchor.shape[0] <= 0:
                    print(f"just after prune, current {self.obj_class} is not ok, {self.model_name}")

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

        if self.use_lod_bias and hasattr(self, '_lod_bias') and self._lod_bias is not None:

            if self._lod_bias.shape[0] == valid_points_mask.shape[0]:

                self._lod_bias = nn.Parameter(self._lod_bias.data[valid_points_mask])
            elif self._lod_bias.shape[0] < valid_points_mask.shape[0]:
                # _lod_biasmask,padding
                padding = torch.zeros(valid_points_mask.shape[0] - self._lod_bias.shape[0],
                                     device=self._lod_bias.device, dtype=self._lod_bias.dtype)
                lod_bias_full = torch.cat([self._lod_bias.data, padding], dim=0)
                self._lod_bias = nn.Parameter(lod_bias_full[valid_points_mask])
            else:
                # _lod_biasmask,
                lod_bias_truncated = self._lod_bias.data[:valid_points_mask.shape[0]]
                self._lod_bias = nn.Parameter(lod_bias_truncated[valid_points_mask])

            # optimizer
            for group in self.optimizer.param_groups:
                if group.get('name') == 'lod_bias':
                    group['params'] = [self._lod_bias]
                    break

    def anchor_growing(self, iteration, grads, threshold, update_ratio, extra_ratio, extra_up, offset_mask, viewpoint_camera):
        init_length = self.get_anchor.shape[0]
        grads[~offset_mask] = 0.0
        anchor_grads = torch.sum(grads.reshape(-1, self.n_offsets), dim=-1) / (
                    torch.sum(offset_mask.reshape(-1, self.n_offsets), dim=-1) + 1e-6)
        for cur_level in range(self.levels):
            update_value = self.fork ** update_ratio
            level_mask = (self.get_level == cur_level).squeeze(dim=1)
            level_ds_mask = (self.get_level == cur_level + 1).squeeze(dim=1)
            if torch.sum(level_mask) == 0:
                continue
            cur_size = self.voxel_size / (float(self.fork) ** cur_level)
            ds_size = cur_size / self.fork
            # update threshold
            cur_threshold = threshold * (update_value ** cur_level)
            ds_threshold = cur_threshold * update_value
            extra_threshold = cur_threshold * extra_ratio
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold) & (grads < ds_threshold)
            candidate_ds_mask = (grads >= ds_threshold)
            candidate_extra_mask = (anchor_grads >= extra_threshold)

            length_inc = self._anchor.shape[0] - init_length
            if length_inc > 0:
                candidate_mask = torch.cat(
                    [candidate_mask, torch.zeros(length_inc * self.n_offsets, dtype=torch.bool, device='cuda')], dim=0)
                candidate_ds_mask = torch.cat(
                    [candidate_ds_mask, torch.zeros(length_inc * self.n_offsets, dtype=torch.bool, device='cuda')],
                    dim=0)
                candidate_extra_mask = torch.cat(
                    [candidate_extra_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

            repeated_mask = repeat(level_mask, 'n -> (n k)', k=self.n_offsets)
            candidate_mask = torch.logical_and(candidate_mask, repeated_mask)
            candidate_ds_mask = torch.logical_and(candidate_ds_mask, repeated_mask)
            if ~self.progressive or iteration > self.coarse_intervals[-1]:
                self._extra_level += extra_up * candidate_extra_mask.float()
            single_rot = getattr(self, 'obj_rot_single', self.obj_rots[0] if hasattr(self, 'obj_rots') and self.obj_rots is not None else torch.zeros(4, device='cuda'))
            single_trans = getattr(self, 'obj_trans_single', self.obj_trans[0] if hasattr(self, 'obj_trans') and self.obj_trans is not None else torch.zeros(3, device='cuda'))
            obj_rot = single_rot.expand(self.get_anchor.shape[0], -1)
            obj_rot = quaternion_to_matrix(obj_rot)
            obj_trans = single_trans.unsqueeze(0).expand(self.get_anchor.shape[0], -1)
            temp = torch.einsum('bij, bj -> bi', obj_rot, self.get_anchor) + obj_trans
            all_xyz = temp.unsqueeze(dim=1) + self._offset * self.get_scaling[:, :3].unsqueeze(dim=1)

            grid_coords = torch.round((temp[level_mask] - self.init_pos) / cur_size).int()
            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round((selected_xyz - self.init_pos) / cur_size).int()
            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True,
                                                                        dim=0)
            if selected_grid_coords_unique.shape[0] > 0 and grid_coords.shape[0] > 0:
                remove_duplicates = self.get_remove_duplicates(grid_coords, selected_grid_coords_unique)
                remove_duplicates = ~remove_duplicates
                candidate_anchor = selected_grid_coords_unique[remove_duplicates] * cur_size + self.init_pos
                new_level = torch.ones(candidate_anchor.shape[0], dtype=torch.int, device='cuda') * cur_level
                candidate_anchor, new_level, _, weed_mask = self.weed_out(candidate_anchor, new_level)
                remove_duplicates_clone = remove_duplicates.clone()
                remove_duplicates[remove_duplicates_clone] = weed_mask
            else:
                candidate_anchor = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates = torch.ones([0], dtype=torch.bool, device='cuda')
                new_level = torch.zeros([0], dtype=torch.int, device='cuda')

            if (~self.progressive or iteration > self.coarse_intervals[-1]) and cur_level < self.levels - 1:
                grid_coords_ds = torch.round((temp[level_ds_mask] - self.init_pos) / ds_size).int()
                selected_xyz_ds = all_xyz.view([-1, 3])[candidate_ds_mask]
                selected_grid_coords_ds = torch.round((selected_xyz_ds - self.init_pos) / ds_size).int()
                selected_grid_coords_unique_ds, inverse_indices_ds = torch.unique(selected_grid_coords_ds,
                                                                            return_inverse=True, dim=0)
                if selected_grid_coords_unique_ds.shape[0] > 0 and grid_coords_ds.shape[0] > 0:
                    remove_duplicates_ds = self.get_remove_duplicates(grid_coords_ds, selected_grid_coords_unique_ds)
                    remove_duplicates_ds = ~remove_duplicates_ds
                    candidate_anchor_ds = selected_grid_coords_unique_ds[remove_duplicates_ds] * ds_size + self.init_pos
                    new_level_ds = torch.ones(candidate_anchor_ds.shape[0], dtype=torch.int, device='cuda') * (
                                cur_level + 1)
                    candidate_anchor_ds, new_level_ds, _, weed_ds_mask = self.weed_out(candidate_anchor_ds,
                                                                                        new_level_ds)
                    remove_duplicates_ds_clone = remove_duplicates_ds.clone()
                    remove_duplicates_ds[remove_duplicates_ds_clone] = weed_ds_mask
                else:
                    candidate_anchor_ds = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                    remove_duplicates_ds = torch.ones([0], dtype=torch.bool, device='cuda')
                    new_level_ds = torch.zeros([0], dtype=torch.int, device='cuda')
            else:
                candidate_anchor_ds = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates_ds = torch.ones([0], dtype=torch.bool, device='cuda')
                new_level_ds = torch.zeros([0], dtype=torch.int, device='cuda')

            if candidate_anchor.shape[0] + candidate_anchor_ds.shape[0] > 0:
                new_anchor = torch.cat([candidate_anchor, candidate_anchor_ds], dim=0)
                new_level = torch.cat([new_level, new_level_ds]).unsqueeze(dim=1).float().cuda()

                candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
                selected_anchor_indices = candidate_indices // self.n_offsets
                new_feat = self._anchor_feat[selected_anchor_indices]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][
                    remove_duplicates]
                new_feat_ds = torch.zeros([candidate_anchor_ds.shape[0], self.feat_dim], dtype=torch.float,
                                            device='cuda')
                new_feat = torch.cat([new_feat, new_feat_ds], dim=0)

                new_scaling = torch.ones_like(candidate_anchor).repeat([1, 2]).float().cuda() * cur_size
                new_scaling_ds = torch.ones_like(candidate_anchor_ds).repeat([1, 2]).float().cuda() * ds_size
                new_scaling = torch.cat([new_scaling, new_scaling_ds], dim=0)
                new_scaling = torch.log(new_scaling)

                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], dtype=torch.float, device='cuda')
                new_rotation_ds = torch.zeros([candidate_anchor_ds.shape[0], 4], dtype=torch.float, device='cuda')
                new_rotation = torch.cat([new_rotation, new_rotation_ds], dim=0)
                new_rotation[:, 0] = 1.0

                new_opacities = inverse_sigmoid(
                    0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))
                new_opacities_ds = inverse_sigmoid(
                    0.1 * torch.ones((candidate_anchor_ds.shape[0], 1), dtype=torch.float, device="cuda"))
                new_opacities = torch.cat([new_opacities, new_opacities_ds], dim=0)

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat(
                    [1, self.n_offsets, 1]).float().cuda()
                new_offsets_ds = torch.zeros_like(candidate_anchor_ds).unsqueeze(dim=1).repeat(
                    [1, self.n_offsets, 1]).float().cuda()
                new_offsets = torch.cat([new_offsets, new_offsets_ds], dim=0)

                new_extra_level = torch.zeros(candidate_anchor.shape[0], dtype=torch.float, device='cuda')
                new_extra_level_ds = torch.zeros(candidate_anchor_ds.shape[0], dtype=torch.float, device='cuda')
                new_extra_level = torch.cat([new_extra_level, new_extra_level_ds])

                new_semantic = torch.zeros((candidate_anchor.shape[0], 1), dtype=torch.float, device='cuda')
                new_semantic_ds = torch.zeros((candidate_anchor_ds.shape[0], 1), dtype=torch.float, device='cuda')
                new_semantic = torch.cat([new_semantic, new_semantic_ds])

                d = {
                    "anchor": new_anchor,
                    "scaling": new_scaling,
                    "rotation": new_rotation,
                    "anchor_feat": new_feat,
                    "offset": new_offsets,
                    "opacity": new_opacities,
                    "semantic": new_semantic,
                    # "f_dc": new_features_dc,
                    # "f_rest": new_features_rest,
                }

                temp_anchor_demon = torch.cat(
                    [self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat(
                    [self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.opacity_accum
                self.opacity_accum = temp_opacity_accum

                torch.cuda.empty_cache()

                optimizable_tensors = self.cat_tensors_to_optimizer(d)
                self._anchor = optimizable_tensors["anchor"]
                self._scaling = optimizable_tensors["scaling"]
                self._rotation = optimizable_tensors["rotation"]
                self._anchor_feat = optimizable_tensors["anchor_feat"]
                self._offset = optimizable_tensors["offset"]
                self._opacity = optimizable_tensors["opacity"]
                self._semantic = optimizable_tensors["semantic"]
                self._level = torch.cat([self._level, new_level], dim=0)
                self._extra_level = torch.cat([self._extra_level, new_extra_level], dim=0)

                if self.use_lod_bias and hasattr(self, '_lod_bias') and self._lod_bias is not None:
                    new_lod_bias = torch.zeros(new_anchor.shape[0], device='cuda', dtype=torch.float32)
                    self._lod_bias = nn.Parameter(torch.cat([self._lod_bias.data, new_lod_bias], dim=0))
                    # optimizer
                    for group in self.optimizer.param_groups:
                        if group.get('name') == 'lod_bias':
                            group['params'] = [self._lod_bias]
                            break
