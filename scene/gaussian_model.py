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
from utils.general_utils import quaternion_to_matrix
from scene.hash_encoding import HashEncoding

class GaussianModel(nn.Module):

    def setup_functions(self):
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

    def __init__(self, sh_degree, args, model_name='background',
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
        self.model_name = model_name
        self.max_sh_degree = sh_degree
        self.active_sh_degree = 0
        self.num_classes = 1 if args.use_semantic else 0
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
        self.add_cov_dist = add_cov_dist
        self.add_color_dist = add_color_dist
        self.add_level = add_level
        self.progressive = progressive
        # Hash encoding config (Instant-NGP style)
        self.use_hash_encoding = getattr(args, "use_hash_encoding", True)
        self.hash_levels = getattr(args, "hash_levels", 12)
        self.hash_feat_dim = getattr(args, "hash_features_per_level", 2)
        self.hash_log2_size = getattr(args, "hash_log2_size", 18)
        self.hash_base_resolution = getattr(args, "hash_base_resolution", 8)
        self.hash_finest_resolution = getattr(args, "hash_finest_resolution", 256)
        # hash encoding,LOD(anchor,hash)
        self.hash_disable_lod = getattr(args, "hash_disable_lod", True)
        self.lod_depth_near = getattr(args, "lod_depth_near", 20.0)
        self.lod_depth_mid = getattr(args, "lod_depth_mid", 40.0)
        self.lod_depth_bias_mid = getattr(args, "lod_depth_bias_mid", 1.0)
        self.lod_depth_bias_far = getattr(args, "lod_depth_bias_far", 2.0)
        self.use_hash_feat_single_level = getattr(args, "use_hash_feat_single_level", False)
        # LOD
        #  hash  warmup +  max_level;, lod_warmup_* .
        if self.use_hash_encoding:
            default_warmup_iters = 300
            if getattr(self, 'hash_levels', None) is not None and int(getattr(self, 'hash_levels', 0)) > 0:
                default_warmup_max_level = max(3, min(int(self.hash_levels) - 1, 6))
            else:
                default_warmup_max_level = 3
        else:
            default_warmup_iters = 1000
            default_warmup_max_level = 1
        self.lod_warmup_iters = getattr(args, "lod_warmup_iters", default_warmup_iters)
        self.lod_warmup_max_level = getattr(args, "lod_warmup_max_level", default_warmup_max_level)
        self.anchor_grow_delay = getattr(args, "anchor_grow_delay", 1000)
        self.hash_encoding = None

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
        self.setup_functions()

        self.opacity_dist_dim = 1 if self.add_opacity_dist else 0
        self.cov_dist_dim = 1 if self.add_cov_dist else 0
        self.color_dist_dim = 1 if self.add_color_dist else 0
        self.level_dim = 1 if self.add_level else 0

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

    def eval(self):
        self.mlp_opacity.eval()
        self.mlp_cov.eval()
        self.mlp_color.eval()
        if self.use_feat_bank:
            self.mlp_feature_bank.eval()
        if self.appearance_dim > 0:
            self.embedding_appearance.eval()
        if hasattr(self, 'hash_encoding') and self.hash_encoding is not None:
            self.hash_encoding.eval()
        #  hash_lod_partitioner ()
        if hasattr(self, 'hash_lod_partitioner') and self.hash_lod_partitioner is not None:
            self.hash_lod_partitioner.eval()

    def train(self):
        self.mlp_opacity.train()
        self.mlp_cov.train()
        self.mlp_color.train()
        if self.use_feat_bank:
            self.mlp_feature_bank.train()
        if self.appearance_dim > 0:
            self.embedding_appearance.train()
        if hasattr(self, 'hash_encoding') and self.hash_encoding is not None:
            self.hash_encoding.train()
        #  hash_lod_partitioner ()
        if hasattr(self, 'hash_lod_partitioner') and self.hash_lod_partitioner is not None:
            self.hash_lod_partitioner.train()

    def capture(self):
        return (
            self._anchor,
            self._level,
            self._offset,
            self._local,
            self._scaling,
            self._rotation,
            self._opacity,
            self._features_dc,
            self._features_rest,
            self._normal,
            self.denom,
            self.optimizer.state_dict(),
            self.spatial_lr_scale,
        )

    def state_dict(self, is_final):
        state_dict = {
            'anchor': self._anchor,
            'level': self._level,
            'offset': self._offset,
            'local': self._local,
            'scaling': self._scaling,
            'rotation': self._rotation,
            'opacity': self._opacity,
            'features_dc': self._features_dc,
            'features_rest': self._features_rest,
            'normal': self._normal,
            'semantic': self._semantic,
        }

        if not is_final:
            state_dict_extra = {
                'spatial_lr_scale': self.spatial_lr_scale,
                'denom': self.denom,
                'active_sh_degree': self.active_sh_degree,
                'optimizer': self.optimizer.state_dict(),
            }

            state_dict.update(state_dict_extra)

        return state_dict

    def restore(self, model_args, training_args):
        (self.active_sh_degree,
        self._anchor,
        self._level,
        self._offset,
        self._local,
        self._scaling,
        self._rotation,
        self._opacity,
        self._normal,

        denom,
        opt_dict,
        self.spatial_lr_scale) = model_args
        self.training_setup(training_args)
        self.denom = denom
        self.optimizer.load_state_dict(opt_dict)


    @property
    def get_appearance(self):
        return self.embedding_appearance

    @property
    def get_scaling(self):
        return 1.0*self.scaling_activation(self._scaling)

    @property
    def get_rotation(self):
        return self.rotation_activation(self._rotation)

    @property
    def get_anchor(self):
        return self._anchor

    @property
    def get_offset(self):
        return self._offset

    @property
    def get_level(self):
        return self._level

    @property
    def get_extra_level(self):
        return self._extra_level

    def get_features(self):
        features_dc = self._features_dc
        features_rest = self._features_rest
        return torch.cat((features_dc, features_rest), dim=1)

    @property
    def get_opacity(self):
        return self.opacity_activation(self._opacity)

    @property
    def get_anchor_feat(self):
        return self._anchor_feat

    def query_hash_feat_single_level(self, anchor: torch.Tensor, level: torch.Tensor):
        """
         anchor  level  hash .
         [N, feat_dim]( hash_feat_dim != feat_dim,);
         None.
        """
        if not self.use_hash_encoding or not self.use_hash_feat_single_level:
            return None

        #  hash encoding
        if self.hash_encoding is None:
            xyz_min = torch.min(anchor.detach(), dim=0).values
            xyz_max = torch.max(anchor.detach(), dim=0).values
            self.init_hash_encoding(xyz_min, xyz_max)

        # level  [N,1]  [N]
        level_int = level.squeeze(-1)
        if getattr(self, 'hash_levels', None) is not None and int(getattr(self, 'hash_levels', 0)) > 0:
            level_int = torch.clamp(level_int, max=int(self.hash_levels) - 1)

        out = torch.zeros(anchor.shape[0], self.hash_feat_dim, device=anchor.device, dtype=anchor.dtype)
        unique_levels = torch.unique(level_int)
        for lvl in unique_levels:
            mask = level_int == lvl
            if not torch.any(mask):
                continue
            grid_coords, _ = self.hash_encoding.quantize_coords(anchor[mask], int(lvl))
            idx = self.hash_encoding.hash_coords(grid_coords)
            out[mask] = self.hash_encoding.tables[int(lvl)](idx)
        if getattr(self, "hash_feat_proj", None) is None and self.hash_feat_dim != self.feat_dim:
            self.hash_feat_proj = nn.Linear(self.hash_feat_dim, self.feat_dim, bias=False).to(device=anchor.device)
        if getattr(self, "hash_feat_proj", None) is not None:
            out = self.hash_feat_proj(out)
        return out

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
        return self.mlp_feature_bank

    def set_appearance(self, num_cameras):
        if self.appearance_dim > 0:
            self.embedding_appearance = Embedding(num_cameras, self.appearance_dim).cuda()

    def get_covariance(self, scaling_modifier = 1):
        return self.covariance_activation(self.get_scaling, scaling_modifier, self._rotation)
    def oneupSHdegree(self):
        if self.active_sh_degree < self.max_sh_degree:
            self.active_sh_degree += 1

    def set_coarse_interval(self, coarse_iter, coarse_factor):
        self.coarse_intervals = []
        # progressive LOD :
        # - LOD tree  self.levels
        # - hash  hash_levels ," level ", iter
        effective_levels = self.levels
        if getattr(self, "use_hash_encoding", False) and getattr(self, "hash_levels", None) is not None and int(getattr(self, "hash_levels", 0)) > 0:
            effective_levels = min(int(self.levels), int(self.hash_levels))
        effective_init_level = min(int(self.init_level), max(effective_levels - 1, 0))
        num_level = effective_levels - 1 - effective_init_level
        if num_level > 0:
            q = 1/coarse_factor
            a1 = coarse_iter*(1-q)/(1-q**num_level)
            temp_interval = 0
            for i in range(num_level):
                interval = a1 * q ** i + temp_interval
                temp_interval = interval
                self.coarse_intervals.append(interval)

    def set_level(self, points, cameras, scales, dist_ratio=0.95, init_level=-1, levels=-1):
        all_dist = torch.tensor([]).cuda()
        self.cam_infos = torch.empty(0, 4).float().cuda()
        for scale in scales:
            for cam in cameras[scale]:
                cam_center = cam.camera_center
                cam_info = torch.tensor([cam_center[0], cam_center[1], cam_center[2], scale]).float().cuda()
                self.cam_infos = torch.cat((self.cam_infos, cam_info.unsqueeze(dim=0)), dim=0)
                dist = torch.sqrt(torch.sum((points - cam_center)**2, dim=1)) * scale
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

            if getattr(self, "use_hash_encoding", False) and getattr(self, "hash_levels", None) is not None and int(getattr(self, "hash_levels")) > 0:
                self.levels = int(self.hash_levels)
            else:
                self.levels = torch.round(torch.log2(dist_max/dist_min)/math.log2(self.fork)).int().item() + 1
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
        print(f"[{getattr(self, 'model_name', 'unknown')}] Building LOD tree time: {int(time_diff // 60)} min {time_diff % 60} sec")

    # =================== Hash encoding helpers ===================
    def init_hash_encoding(self, xyz_min: torch.Tensor, xyz_max: torch.Tensor, n_levels: int = None):
        """Initialize hash encoding with current bounding box.

        If `n_levels` is provided and positive, use it; otherwise fall back to `self.hash_levels`.
        """
        use_levels = None
        if n_levels is not None and int(n_levels) > 0:
            use_levels = int(n_levels)
        elif getattr(self, 'hash_levels', None) is not None and int(getattr(self, 'hash_levels', 0)) > 0:
            use_levels = int(self.hash_levels)
        else:
            # Fallback to a reasonable default if nothing specified
            use_levels = 8

        self.hash_encoding = HashEncoding(
            n_levels=use_levels,
            n_features_per_level=self.hash_feat_dim,
            log2_hashmap_size=self.hash_log2_size,
            base_resolution=self.hash_base_resolution,
            finest_resolution=self.hash_finest_resolution,
            aabb=(xyz_min, xyz_max),
        ).cuda()

    def hash_sample(self, data: torch.Tensor, xyz_min: torch.Tensor, xyz_max: torch.Tensor, base_voxel_size=None, levels=None):
        """
        Hash-based sampling to generate anchors and levels (LOD).

        ( LOD tree ):
        1. , anchor  LOD tree
        2.  LOD tree  default_voxel_size (0.02)
        3. fork=2 ()
        """
        torch.cuda.synchronize()
        t0 = time.time()

        # ==================== lod/voxel_size  =====================

        if base_voxel_size is not None and levels is not None:
            #  LOD tree
            total_levels = int(levels)
            base_voxel_size = float(base_voxel_size)
            print(f"[hash_sample] Using externally provided base_voxel_size={base_voxel_size}, levels={total_levels}")
            self.levels = total_levels
            #  max_range  hash resolution
            data_min = torch.min(data, dim=0).values
            data_max = torch.max(data, dim=0).values
            data_range = data_max - data_min
            max_range = torch.max(data_range).item()
            default_voxel_size = 0.02  # , LOD tree
            fork = float(self.fork) if hasattr(self, 'fork') else 2.0
        else:

            data_min = torch.min(data, dim=0).values
            data_max = torch.max(data, dim=0).values
            data_range = data_max - data_min
            max_range = torch.max(data_range).item()

            #  LOD tree
            default_voxel_size = 0.02  # , LOD tree
            fork = float(self.fork) if hasattr(self, 'fork') else 2.0

            # ( LOD tree  base_layer )
            adaptive_levels = int(torch.round(torch.log2(torch.tensor(max_range / default_voxel_size))).item())

            # (hash_levels  None  <=0)( pre_hash_base_voxel).
            config_val = getattr(self, 'hash_levels', None)
            pre_base = getattr(self, 'pre_hash_base_voxel', None)

            #  scalar_pre  computed_pre_levels,
            scalar_pre = None
            computed_pre_levels = None
            if pre_base is not None:
                try:
                    if isinstance(pre_base, torch.Tensor):
                        scalar_pre = float(pre_base.max().item())
                    else:
                        scalar_pre = float(pre_base)
                except Exception:
                    try:
                        scalar_pre = float(pre_base) if not isinstance(pre_base, torch.Tensor) else float(pre_base.max().item())
                    except Exception:
                        scalar_pre = None

            if config_val is None or int(config_val) <= 0:

                if scalar_pre is not None:
                    #  pre_base  levels()
                    try:
                        computed_pre_levels = int(round(math.log(scalar_pre / default_voxel_size, fork))) + 1
                    except Exception:
                        computed_pre_levels = adaptive_levels
                    total_levels = max(4, computed_pre_levels)
                    used_reason = f'computed_from_pre({computed_pre_levels})'
                else:
                    total_levels = max(4, adaptive_levels)
                    used_reason = 'adaptive'
                config_levels = 'adaptive'
            else:
                # ,( adaptive/min )
                total_levels = int(config_val)
                used_reason = f'user_specified({total_levels})'
                config_levels = int(config_val)

                #  total_levels ( anchors )
                try:
                    total_levels = int(total_levels)
                except Exception:
                    total_levels = max(4, int(adaptive_levels))
                # clamp : 4, 12(/)
                total_levels = max(4, min(total_levels, 12))

            # ()-- LOD tree
            # (max_range) default_voxel_size  base_layer,
            #  hash  anchor .
            print(f"[hash_sample] total_levels: {total_levels}")
            try:
                base_layer = int(torch.round(torch.log2(torch.tensor(max_range / default_voxel_size))).item()) - (total_levels // 2) + 1
            except Exception:
                base_layer = max(0, total_levels - 1)
            if base_layer < 0:
                base_layer = 0

            # 1: total_levels , default_voxel_size, anchor .
            if base_voxel_size is None:
                base_voxel_size = default_voxel_size * (fork ** (total_levels - 1))
            print(f"[hash_sample] base_layer: {base_layer}, base_voxel_size: {base_voxel_size}")

            print(f"[{getattr(self, 'model_name', 'unknown')}] hash_sample :")
            print(f"  - data range: {data_range.tolist()}, max_range={max_range:.4f}")
            print(f"  - levels: {total_levels} (config={config_levels}, adaptive={adaptive_levels}, reason={used_reason})")
            if pre_base is not None:
                print(f"  - pre_hash_base_voxel(reference): {scalar_pre:.4f}, computed_pre_levels: {computed_pre_levels}")
            print(f"  - base voxel size: {base_voxel_size:.6f}, finest voxel size: {base_voxel_size / (fork ** (total_levels - 1)):.6f}")
        self.levels = total_levels

        # ====================  hash resolution () ====================
        #  hash_base_resolution/hash_finest_resolution, desired finest voxel
        #  Instant-NGP :r^(l) = floor(r_base * b^{l/(L-1)}), b ~ 2
        r_finest = int(math.ceil(max_range / default_voxel_size))

        if not hasattr(self, 'hash_finest_resolution') or getattr(self, 'hash_finest_resolution', None) is None or int(getattr(self, 'hash_finest_resolution', 0)) <= 0:
            self.hash_finest_resolution = r_finest
        if not hasattr(self, 'hash_base_resolution') or getattr(self, 'hash_base_resolution', None) is None or int(getattr(self, 'hash_base_resolution', 0)) <= 0:
            #  base_resolution, base -> finest  2^ (L-1)
            base_res = max(16, int(max(1, math.floor(self.hash_finest_resolution / (2 ** (total_levels - 1))))))
            self.hash_base_resolution = base_res

        # Keep hash-table levels independent from anchor sampling levels.
        hash_encoding_levels = None
        if getattr(self, 'hash_levels', None) is not None and int(getattr(self, 'hash_levels', 0)) > 0:
            hash_encoding_levels = int(self.hash_levels)

        #  hash encoding( total_levels)
        if self.hash_encoding is None:
            self.init_hash_encoding(xyz_min, xyz_max, n_levels=hash_encoding_levels)

        # ==================== ( LOD tree )====================
        positions = []
        levels = []

        for cur_level in range(total_levels):
            #  LOD tree
            cur_size = base_voxel_size / (fork ** cur_level)

            #  lod_tree_sample :
            quantized = torch.round((data - data_min) / cur_size) * cur_size + data_min
            unique_pos = torch.unique(quantized, dim=0)

            positions.append(unique_pos)
            level_tensor = torch.full((unique_pos.shape[0],), cur_level, dtype=torch.int, device=data.device)
            levels.append(level_tensor)

            if cur_level < 3 or cur_level == total_levels - 1:
                print(f"  - Level {cur_level}: voxel_size={cur_size:.4f}, anchors={unique_pos.shape[0]}")

        self.positions = torch.cat(positions, dim=0)
        self._level = torch.cat(levels, dim=0).int()

        #  voxel_size ( LOD tree )
        self.voxel_size = torch.tensor(base_voxel_size, device=data.device, dtype=torch.float32)


        pos_min = torch.min(self.positions, dim=0).values
        pos_max = torch.max(self.positions, dim=0).values
        print(f"[{getattr(self, 'model_name', 'unknown')}] hash_sample result: {self.positions.shape[0]} anchors")
        print(f"  - Position range: x=[{pos_min[0].item():.2f}, {pos_max[0].item():.2f}], "
              f"y=[{pos_min[1].item():.2f}, {pos_max[1].item():.2f}], "
              f"z=[{pos_min[2].item():.2f}, {pos_max[2].item():.2f}]")

        torch.cuda.synchronize()
        t1 = time.time()
        time_diff = t1 - t0
        print(f"[{getattr(self, 'model_name', 'unknown')}] Building hash grid time: {int(time_diff // 60)} min {time_diff % 60:.2f} sec")

    def create_from_pcd(self, points: BasicPointCloud, spatial_lr_scale: float, logger=None):
        self.spatial_lr_scale = spatial_lr_scale
        xyz_min = torch.min(points, dim=0).values * self.extend
        xyz_max = torch.max(points, dim=0).values * self.extend
        box_d = xyz_max - xyz_min
        if self.base_layer < 0:
            default_voxel_size = 0.02
            scalar_box = torch.max(box_d)
            self.base_layer = int(torch.round(torch.log2(scalar_box/default_voxel_size)).item())-(self.levels//2)+1
        self.voxel_size = box_d/(float(self.fork) ** self.base_layer)
        self.init_pos = xyz_min.clone().float().cuda()

        used_hash = bool(getattr(self, 'use_hash_encoding', False))
        pre_voxel_size = self.voxel_size.clone() if isinstance(self.voxel_size, torch.Tensor) else self.voxel_size
        try:
            self.pre_hash_base_voxel = pre_voxel_size.max().detach().cpu() if isinstance(pre_voxel_size, torch.Tensor) else float(pre_voxel_size)
        except Exception:
            self.pre_hash_base_voxel = float(pre_voxel_size) if not isinstance(pre_voxel_size, torch.Tensor) else float(pre_voxel_size.max().item())
        print(f"[{getattr(self, 'model_name', 'unknown')}] init sampling: use_hash_encoding={used_hash}")
        if used_hash:
            #  hash ,hash_sample  self.voxel_size
            self.hash_sample(points, xyz_min, xyz_max)
        else:
            self.lod_tree_sample(points, self.init_pos)

        if self.visible_threshold < 0:
            self.visible_threshold = 0.0
            self.positions, self._level, self.visible_threshold, _ = self.weed_out(self.positions, self._level)
        self.positions, self._level, _, _ = self.weed_out(self.positions, self._level)

        print(f'Branches of Tree: {self.fork}')
        print(f'Use Hash Encoding: {used_hash}')
        if used_hash:
            print(f'Pre-hash Base Layer: {self.base_layer} (computed from box)')
            print(f'Pre-hash Voxel Size: {pre_voxel_size}')
            print(f'Hash Params: hash_levels={getattr(self, "hash_levels", None)}, hash_base_resolution={getattr(self, "hash_base_resolution", None)}, hash_finest_resolution={getattr(self, "hash_finest_resolution", None)}, hash_log2_size={getattr(self, "hash_log2_size", None)}')
            print(f'LOD Levels (final): {self.levels}')
            print(f'Final Voxel Size (post-hash): {self.voxel_size}')
        else:
            print(f'Base Layer of Tree: {self.base_layer}')
            print(f'normal LOD Levels: {self.levels}')
            print(f'Max Voxel Size: {self.voxel_size}')
            print(f'Min Voxel Size: {self.voxel_size/(2.0 ** (self.levels - 1))}')

        print(f'Visible Threshold: {self.visible_threshold}')
        print(f'Appearance Embedding Dimension: {self.appearance_dim}')
        print(f'Initial Levels: {self.init_level}')
        print(f'Initial Voxel Number: {self.positions.shape[0]}')

        logger.info(f'Branches of Tree: {self.fork}')
        logger.info(f'Use Hash Encoding: {used_hash}')
        if used_hash:
            logger.info(f'Pre-hash Base Layer: {self.base_layer} (computed from box)')
            logger.info(f'Pre-hash Voxel Size: {pre_voxel_size}')
            logger.info(f'Hash Params: hash_levels={getattr(self, "hash_levels", None)}, hash_base_resolution={getattr(self, "hash_base_resolution", None)}, hash_finest_resolution={getattr(self, "hash_finest_resolution", None)}, hash_log2_size={getattr(self, "hash_log2_size", None)}')
            logger.info(f'LOD Levels (final): {self.levels}')
            logger.info(f'Final Voxel Size (post-hash): {self.voxel_size}')
        else:
            logger.info(f'Base Layer of Tree: {self.base_layer}')
            logger.info(f'normal LOD Levels: {self.levels}')
            logger.info(f'Max Voxel Size: {self.voxel_size}')
            logger.info(f'Min Voxel Size: {self.voxel_size/(2.0 ** (self.levels - 1))}')

        offsets = torch.zeros((self.positions.shape[0], self.n_offsets, 3)).float().cuda()
        # LOD tree:hash,0
        # hash_lod0,hash
        anchors_feat = torch.zeros((self.positions.shape[0], self.feat_dim)).float().cuda()
        dist2 = torch.clamp_min(distCUDA2(self.positions).float().cuda(), 0.0000001)
        scales = torch.log(torch.sqrt(dist2))[...,None].repeat(1, 6)
        rots = torch.zeros((self.positions.shape[0], 4), device="cuda")
        rots[:, 0] = 1
        opacities = inverse_sigmoid(0.1 * torch.ones((self.positions.shape[0], 1), dtype=torch.float, device="cuda"))
        semamtics = torch.zeros((self.positions.shape[0], self.num_classes), dtype=torch.float, device="cuda")

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


    def map_to_int_level(self, pred_level, cur_level):
        # When hash encoding is enabled, cap by hash_levels-1 to align with hash LOD.
        if self.use_hash_encoding and getattr(self, 'hash_levels', None) is not None and int(getattr(self, 'hash_levels', 0)) > 0:
            cur_level = min(cur_level, int(self.hash_levels) - 1)
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
    #
    #

    def set_anchor_mask_perlevel(self, cam_center, resolution_scale, cur_level):
        anchor_pos = self._anchor + (self.voxel_size/2) / (float(self.fork) ** self._level)
        dist = torch.sqrt(torch.sum((anchor_pos - cam_center)**2, dim=1)) * resolution_scale
        effective_standard_dist = self.standard_dist
        pred_level = torch.log2(effective_standard_dist/dist)/math.log2(self.fork) + self._extra_level
        int_level = self.map_to_int_level(pred_level, cur_level)
        self._anchor_mask = (self._level.squeeze(dim=1) <= int_level)

    def training_setup(self, training_args):
        self.percent_dense = training_args.percent_dense

        self.opacity_accum = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

        self.offset_gradient_accum = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.offset_denom = torch.zeros((self.get_anchor.shape[0]*self.n_offsets, 1), device="cuda")
        self.anchor_demon = torch.zeros((self.get_anchor.shape[0], 1), device="cuda")

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
        if self.appearance_dim > 0:
            l.append({'params': self.embedding_appearance.parameters(), 'lr': training_args.appearance_lr_init, "name": "embedding_appearance"})
        if self.use_feat_bank:
            l.append({'params': self.mlp_feature_bank.parameters(), 'lr': training_args.mlp_featurebank_lr_init, "name": "mlp_featurebank"})

        self.optimizer = torch.optim.Adam(l, lr=0.0, eps=1e-15)
        self.anchor_scheduler_args = get_expon_lr_func(lr_init=training_args.position_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.position_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.position_lr_delay_mult,
                                                    max_steps=training_args.position_lr_max_steps)
        self.offset_scheduler_args = get_expon_lr_func(lr_init=training_args.offset_lr_init*self.spatial_lr_scale,
                                                    lr_final=training_args.offset_lr_final*self.spatial_lr_scale,
                                                    lr_delay_mult=training_args.offset_lr_delay_mult,
                                                    max_steps=training_args.offset_lr_max_steps)

        self.mlp_opacity_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_opacity_lr_init,
                                                    lr_final=training_args.mlp_opacity_lr_final,
                                                    lr_delay_mult=training_args.mlp_opacity_lr_delay_mult,
                                                    max_steps=training_args.mlp_opacity_lr_max_steps)

        self.mlp_cov_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_cov_lr_init,
                                                    lr_final=training_args.mlp_cov_lr_final,
                                                    lr_delay_mult=training_args.mlp_cov_lr_delay_mult,
                                                    max_steps=training_args.mlp_cov_lr_max_steps)

        self.mlp_color_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_color_lr_init,
                                                    lr_final=training_args.mlp_color_lr_final,
                                                    lr_delay_mult=training_args.mlp_color_lr_delay_mult,
                                                    max_steps=training_args.mlp_color_lr_max_steps)
        if self.use_feat_bank:
            self.mlp_featurebank_scheduler_args = get_expon_lr_func(lr_init=training_args.mlp_featurebank_lr_init,
                                                        lr_final=training_args.mlp_featurebank_lr_final,
                                                        lr_delay_mult=training_args.mlp_featurebank_lr_delay_mult,
                                                        max_steps=training_args.mlp_featurebank_lr_max_steps)
        if self.appearance_dim > 0:
            self.appearance_scheduler_args = get_expon_lr_func(lr_init=training_args.appearance_lr_init,
                                                        lr_final=training_args.appearance_lr_final,
                                                        lr_delay_mult=training_args.appearance_lr_delay_mult,
                                                        max_steps=training_args.appearance_lr_max_steps)

    def update_learning_rate(self, iteration):
        ''' Learning rate scheduling per step '''
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "offset":
                lr = self.offset_scheduler_args(iteration)
                param_group['lr'] = lr
            if param_group["name"] == "anchor":
                lr = self.anchor_scheduler_args(iteration)
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

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))
        print(f"Saving ply to {path}")
        anchor = self._anchor.detach().cpu().numpy()
        levels = self._level.detach().cpu().numpy()
        extra_levels = self._extra_level.unsqueeze(dim=1).detach().cpu().numpy()

        # infos,
        infos = np.zeros((anchor.shape[0], 2), dtype=np.float32)

        voxel_size_val = self.voxel_size
        if torch.is_tensor(voxel_size_val):
            voxel_size_val = voxel_size_val.cpu().numpy()
        if isinstance(voxel_size_val, np.ndarray) and voxel_size_val.size > 1:
            voxel_size_val = float(np.mean(voxel_size_val))
        else:
            voxel_size_val = float(voxel_size_val.item() if hasattr(voxel_size_val, 'item') else voxel_size_val)
        infos[0, 0] = voxel_size_val

        # standard_dist:
        standard_dist_val = self.standard_dist
        if torch.is_tensor(standard_dist_val):
            standard_dist_val = standard_dist_val.cpu().numpy()
        if isinstance(standard_dist_val, np.ndarray) and standard_dist_val.size > 1:
            standard_dist_val = float(np.mean(standard_dist_val))
        else:
            standard_dist_val = float(standard_dist_val.item() if hasattr(standard_dist_val, 'item') else standard_dist_val)
        infos[1, 0] = standard_dist_val

        anchor_feats = self._anchor_feat.detach().cpu().numpy()
        offsets = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scales = self._scaling.detach().cpu().numpy()
        rots = self._rotation.detach().cpu().numpy()
        semantic = self._semantic.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(anchor.shape[0], dtype=dtype_full)

        attributes = np.concatenate((anchor, levels, extra_levels, infos, offsets, anchor_feats, opacities, scales, rots, semantic), axis=1)
        elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def save_ply_at_t(self, path, t=0.0):
        """,

        Args:
            path:
            t: ,,
        """
        mkdir_p(os.path.dirname(path))
        print(f" {path}")

        # PLY
        self.save_ply(path)


        ellipsoid_dir = os.path.join(os.path.dirname(path), "ellipsoids")
        mkdir_p(ellipsoid_dir)


        self.save_ellipsoid_mesh(ellipsoid_dir)

        # PLY
        self.visualize_gaussian_ellipsoids_simple(ellipsoid_dir)


        self.save_colored_points(ellipsoid_dir)

        print(f"Ellipsoid visualization saved to: {ellipsoid_dir}")
        return ellipsoid_dir

    def save_ellipsoid_mesh(self, save_dir):
        """

        Args:
            save_dir:
        """
        try:
            import numpy as np
            import open3d as o3d
            from scipy.spatial.transform import Rotation
            import os

            print("Saving Gaussian ellipsoid mesh...")


            points = self.get_anchor.detach().cpu().numpy()
            scales = self.get_scaling.detach().cpu().numpy()


            if scales.shape[1] > 3:
                scales = scales[:, :3]  #  (xyz)


            has_rotation = False
            try:
                rotations = self.get_rotation.detach().cpu().numpy()
                has_rotation = True
                print("Rotation data found.")
            except:
                print("Rotation data unavailable; using axis-aligned ellipsoids.")


            MAX_ELLIPSOIDS = 10000
            if len(points) > MAX_ELLIPSOIDS:
                print(f"Subsampling ellipsoids to {MAX_ELLIPSOIDS}.")
                indices = np.random.choice(len(points), MAX_ELLIPSOIDS, replace=False)
                points = points[indices]
                scales = scales[indices]
                if has_rotation:
                    rotations = rotations[indices]


            all_meshes = []


            print(f"Building ellipsoid meshes: {len(points)} points.")


            sphere = o3d.geometry.TriangleMesh.create_sphere(radius=1.0, resolution=10)

            for i in range(len(points)):

                ellipsoid = o3d.geometry.TriangleMesh()
                ellipsoid.vertices = o3d.utility.Vector3dVector(np.asarray(sphere.vertices))
                ellipsoid.triangles = o3d.utility.Vector3iVector(np.asarray(sphere.triangles))


                scale_factors = scales[i]
                ellipsoid.vertices = o3d.utility.Vector3dVector(
                    np.asarray(ellipsoid.vertices) * scale_factors[None, :]
                )


                if has_rotation:
                    try:
                        rot_data = rotations[i]

                        if len(rot_data) == 4:
                            rot_data = rot_data / np.linalg.norm(rot_data)
                            R = Rotation.from_quat(rot_data).as_matrix()
                            ellipsoid.rotate(R)
                        elif len(rot_data) == 9:
                            R = rot_data.reshape(3, 3)
                            ellipsoid.rotate(R)
                    except Exception as e:
                        print(f"Rotation conversion failed: {e}")


                ellipsoid.translate(points[i])


                depth_norm = (points[i, 2] - np.min(points[:, 2])) / (np.max(points[:, 2]) - np.min(points[:, 2]) + 1e-6)
                color = np.array([1-depth_norm, 0.5, depth_norm])
                ellipsoid.paint_uniform_color(color)

                all_meshes.append(ellipsoid)


                if (i+1) % 1000 == 0:
                    print(f" {i+1}/{len(points)} ")


            if all_meshes:
                print("Combining ellipsoid meshes...")
                combined_mesh = all_meshes[0]
                for mesh in all_meshes[1:]:
                    combined_mesh += mesh


                combined_mesh.compute_vertex_normals()

                # PLY
                mesh_path = os.path.join(save_dir, "gaussian_ellipsoids.ply")
                o3d.io.write_triangle_mesh(mesh_path, combined_mesh)
                print(f"Saved ellipsoid mesh: {mesh_path}")


                edges = o3d.geometry.LineSet.create_from_triangle_mesh(combined_mesh)
                edges.paint_uniform_color([0.0, 0.0, 0.0])


                edges_path = os.path.join(save_dir, "gaussian_ellipsoids_wireframe.ply")
                o3d.io.write_line_set(edges_path, edges)
                print(f"Saved ellipsoid wireframe: {edges_path}")

                return True
            else:
                print("No ellipsoid meshes to save.")
                return False

        except ImportError:
            print("Open3D is required to save ellipsoid meshes.")
            return False
        except Exception as e:
            print(f"Failed to save ellipsoid meshes: {e}")
            import traceback
            traceback.print_exc()
            return False

    def visualize_gaussian_ellipsoids_simple(self, save_dir):
        """

        Args:
            save_dir:
        """
        try:
            import numpy as np
            import os
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D

            print("Saving top-view Gaussian point visualization...")


            points = self.get_anchor.detach().cpu().numpy()
            scales = self.get_scaling.detach().cpu().numpy()


            MAX_POINTS = 5000
            if len(points) > MAX_POINTS:
                print(f"Subsampling Gaussian points to {MAX_POINTS}.")
                indices = np.random.choice(len(points), MAX_POINTS, replace=False)
                points = points[indices]
                scales = scales[indices]


            min_bound = np.min(points, axis=0)
            max_bound = np.max(points, axis=0)
            center = (min_bound + max_bound) / 2
            size = np.max(max_bound - min_bound)


            angles = [0, 45, 90, 135]

            for i, angle in enumerate(angles):

                fig = plt.figure(figsize=(10, 8), dpi=100)
                ax = fig.add_subplot(111, projection='3d')


                if scales.shape[1] >= 3:
                    sizes = np.mean(scales[:, :3], axis=1) * 100
                else:
                    sizes = scales[:, 0] * 100
                sizes = np.clip(sizes, 5, 500)


                normalized_z = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2]) - np.min(points[:, 2]) + 1e-8)
                colors = plt.cm.jet(normalized_z)


                ax.scatter(
                    points[:, 0], points[:, 1], points[:, 2],
                    s=sizes,
                    c=colors,
                    alpha=0.6,
                    edgecolors='none'
                )


                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')


                ax.view_init(elev=30, azim=angle)


                ax.set_xlim(center[0] - size/2, center[0] + size/2)
                ax.set_ylim(center[1] - size/2, center[1] + size/2)
                ax.set_zlim(center[2] - size/2, center[2] + size/2)


                ax.set_facecolor('white')


                ax.grid(True, linestyle='--', alpha=0.6)


                save_path = os.path.join(save_dir, f"gaussian_points_view_{i}.png")
                plt.tight_layout()
                plt.savefig(save_path, bbox_inches='tight')
                plt.close()

                print(f" {i+1}/{len(angles)} : {save_path}")


            fig = plt.figure(figsize=(10, 8), dpi=100)
            ax = fig.add_subplot(111)


            normalized_z = (points[:, 2] - np.min(points[:, 2])) / (np.max(points[:, 2]) - np.min(points[:, 2]) + 1e-8)
            colors = plt.cm.jet(normalized_z)


            sc = ax.scatter(
                points[:, 0], points[:, 1],
                s=sizes/2,
                c=colors,
                alpha=0.6,
                edgecolors='none'
            )


            cbar = plt.colorbar(sc)
            cbar.set_label('Z ')


            ax.set_title('')
            ax.set_xlabel('X')
            ax.set_ylabel('Y')


            ax.set_xlim(center[0] - size/2, center[0] + size/2)
            ax.set_ylim(center[1] - size/2, center[1] + size/2)


            ax.grid(True, linestyle='--', alpha=0.6)


            save_path = os.path.join(save_dir, "gaussian_points_top_view.png")
            plt.tight_layout()
            plt.savefig(save_path, bbox_inches='tight')
            plt.close()

            print(f"Saved top-view Gaussian points: {save_path}")
            return True

        except ImportError:
            print("matplotlib is required to visualize Gaussian ellipsoids.")
            return False
        except Exception as e:
            print(f"Failed to visualize Gaussian ellipsoids: {e}")
            import traceback
            traceback.print_exc()
            return False

    def save_colored_points(self, save_dir):
        """

        Args:
            save_dir:
        """
        try:
            import numpy as np
            import os
            import open3d as o3d

            print("Saving colored Gaussian points...")


            points = self.get_anchor.detach().cpu().numpy()
            scales = self.get_scaling.detach().cpu().numpy()

            # Open3D
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)


            colors = np.zeros((len(points), 3))
            if scales.shape[1] >= 3:
                scale_magnitude = np.mean(scales[:, :3], axis=1)
            else:
                scale_magnitude = scales[:, 0]

            normalized_scale = (scale_magnitude - np.min(scale_magnitude)) / (np.max(scale_magnitude) - np.min(scale_magnitude) + 1e-6)


            for i in range(len(points)):
                t = normalized_scale[i]
                if t < 0.25:
                    colors[i] = [0, 0, 1.0]
                elif t < 0.5:
                    colors[i] = [0, 1.0, 0]
                elif t < 0.75:
                    colors[i] = [1.0, 1.0, 0]
                else:
                    colors[i] = [1.0, 0, 0]

            pcd.colors = o3d.utility.Vector3dVector(colors)


            o3d.io.write_point_cloud(os.path.join(save_dir, "gaussian_points_colored.ply"), pcd)
            print(f"Saved colored Gaussian points: {os.path.join(save_dir, 'gaussian_points_colored.ply')}")
            return True

        except ImportError:
            print("Open3D is required to save colored Gaussian points.")
            return False
        except Exception as e:
            print(f"Failed to save colored Gaussian points: {e}")
            import traceback
            traceback.print_exc()
            return False

    def make_ply(self, path):
        if self._anchor is None or self._anchor.shape[0] == 0:
            return None


        anchor = self._anchor.detach().cpu().numpy()
        levels = self._level.detach().cpu().numpy()
        extra_levels = self._extra_level.unsqueeze(dim=1).detach().cpu().numpy()

        # infos,
        infos = np.zeros((anchor.shape[0], 2), dtype=np.float32)

        voxel_size_val = self.voxel_size
        if torch.is_tensor(voxel_size_val):
            voxel_size_val = voxel_size_val.cpu().numpy()
        if isinstance(voxel_size_val, np.ndarray) and voxel_size_val.size > 1:
            voxel_size_val = float(np.mean(voxel_size_val))
        else:
            voxel_size_val = float(voxel_size_val.item() if hasattr(voxel_size_val, 'item') else voxel_size_val)
        infos[:, 0] = voxel_size_val

        # standard_dist:
        standard_dist_val = self.standard_dist
        if torch.is_tensor(standard_dist_val):
            standard_dist_val = standard_dist_val.cpu().numpy()
        if isinstance(standard_dist_val, np.ndarray) and standard_dist_val.size > 1:
            standard_dist_val = float(np.mean(standard_dist_val))
        else:
            standard_dist_val = float(standard_dist_val.item() if hasattr(standard_dist_val, 'item') else standard_dist_val)
        infos[:, 1] = standard_dist_val

        anchor_feats = self._anchor_feat.detach().cpu().numpy()
        offsets = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scales = self._scaling.detach().cpu().numpy()
        rots = self._rotation.detach().cpu().numpy()
        semantic = self._semantic.detach().cpu().numpy()

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]
        elements = np.empty(anchor.shape[0], dtype=dtype_full)

        attributes = np.concatenate((anchor, levels, extra_levels, infos, offsets, anchor_feats, opacities, scales, rots, semantic), axis=1)
        elements[:] = list(map(tuple, attributes))

        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def load_ply_sparse_gaussian(self, path, plydata):
        print(f"Loading PLY data for model {self.model_name}")


        anchor = np.stack((np.asarray(plydata["x"]),
                        np.asarray(plydata["y"]),
                        np.asarray(plydata["z"])),  axis=1).astype(np.float32)
        print(f"Loaded {anchor.shape[0]} points")

        # tensor,
        anchor = torch.tensor(anchor, dtype=torch.float, device="cuda")

        levels = np.asarray(plydata["level"])[... ,np.newaxis].astype(np.int)
        extra_levels = np.asarray(plydata["extra_level"])[... ,np.newaxis].astype(np.float32)
        self.voxel_size = torch.tensor(plydata["info_0"][0]).float()
        self.standard_dist = torch.tensor(plydata["info_1"][0]).float()

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

        # tensor,
        rots = torch.tensor(rots, dtype=torch.float, device="cuda")

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

        # tensor,
        offsets = torch.tensor(offsets, dtype=torch.float, device="cuda")

        semantic_names = [p.name for p in plydata.properties if p.name.startswith("semantic")]
        semantic_names = sorted(semantic_names, key = lambda x: int(x.split('_')[-1]))
        semantic = np.zeros((anchor.shape[0], len(semantic_names)))
        for idx, attr_name in enumerate(semantic_names):
            semantic[:, idx] = np.asarray(plydata[attr_name]).astype(np.float32)


        self._anchor_feat = nn.Parameter(torch.tensor(anchor_feats, dtype=torch.float, device="cuda").requires_grad_(True))
        self._level = torch.tensor(levels, dtype=torch.int, device="cuda")
        self._extra_level = torch.tensor(extra_levels, dtype=torch.float, device="cuda").squeeze(dim=1)
        self._offset = nn.Parameter(offsets.transpose(1, 2).contiguous().requires_grad_(True))
        self._anchor = nn.Parameter(anchor.requires_grad_(True))
        self._scaling = nn.Parameter(torch.tensor(scales, dtype=torch.float, device="cuda").requires_grad_(True))
        self._opacity = nn.Parameter(torch.tensor(opacities, dtype=torch.float, device="cuda").requires_grad_(False))
        self._semantic = nn.Parameter(torch.tensor(semantic, dtype=torch.float, device="cuda").requires_grad_(True))
        self._rotation = nn.Parameter(rots.requires_grad_(False))
        self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device="cuda")
        self.levels = torch.max(self._level) - torch.min(self._level) + 1

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


    # statis grad information to guide liftting.
    def training_statis(self, viewspace_point_tensor, opacity, update_filter, offset_selection_mask, anchor_visible_mask):
        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0

        temp_opacity = temp_opacity.view([-1, self.n_offsets])
        self.opacity_accum[anchor_visible_mask] += temp_opacity.sum(dim=1, keepdim=True)

        # update anchor visiting statis
        self.anchor_demon[anchor_visible_mask] += 1

        # update neural gaussian statis
        anchor_visible_mask = anchor_visible_mask.unsqueeze(dim=1).repeat([1, self.n_offsets]).view(-1)
        combined_mask = torch.zeros_like(self.offset_gradient_accum, dtype=torch.bool).squeeze(dim=1)
        combined_mask[anchor_visible_mask] = offset_selection_mask
        temp_mask = combined_mask.clone()
        combined_mask[temp_mask] = update_filter

        grad_norm = torch.norm(viewspace_point_tensor.grad[update_filter,:2], dim=-1, keepdim=True)
        self.offset_gradient_accum[combined_mask] += grad_norm
        self.offset_denom[combined_mask] += 1

    def _prune_anchor_optimizer(self, mask):
        optimizable_tensors = {}
        for group in self.optimizer.param_groups:
            if  'mlp' in group['name'] or \
                'conv' in group['name'] or \
                'feat_base' in group['name'] or \
                'embedding' in group['name']:
                continue
            if len(group["params"]) > 1:
                continue

            stored_state = self.optimizer.state.get(group['params'][0], None)
            if stored_state is not None:
                stored_state["exp_avg"] = stored_state["exp_avg"][mask]
                stored_state["exp_avg_sq"] = stored_state["exp_avg_sq"][mask]

                del self.optimizer.state[group['params'][0]]
                group["params"][0] = nn.Parameter((group["params"][0][mask].requires_grad_(True)))
                self.optimizer.state[group['params'][0]] = stored_state
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]
            else:
                group["params"][0] = nn.Parameter(group["params"][0][mask].requires_grad_(True))
                if group['name'] == "scaling":
                    scales = group["params"][0]
                    temp = scales[:,3:]
                    temp[temp>0.05] = 0.05
                    group["params"][0][:,3:] = temp
                optimizable_tensors[group["name"]] = group["params"][0]

        return optimizable_tensors

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
        if use_chunk:
            chunk_size = 4096
            max_iters = grid_coords.shape[0] // chunk_size + (1 if grid_coords.shape[0] % chunk_size != 0 else 0)
            remove_duplicates_list = []
            for i in range(max_iters):
                cur_remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords[i*chunk_size:(i+1)*chunk_size, :]).all(-1).any(-1).view(-1)
                remove_duplicates_list.append(cur_remove_duplicates)
            remove_duplicates = reduce(torch.logical_or, remove_duplicates_list)
        else:
            remove_duplicates = (selected_grid_coords_unique.unsqueeze(1) == grid_coords).all(-1).any(-1).view(-1)
        return remove_duplicates

    def anchor_growing(self, iteration, grads, threshold, update_ratio, extra_ratio, extra_up, offset_mask):
        init_length = self.get_anchor.shape[0]
        grads[~offset_mask] = 0.0
        anchor_grads = torch.sum(grads.reshape(-1, self.n_offsets), dim=-1) / (torch.sum(offset_mask.reshape(-1, self.n_offsets), dim=-1) + 1e-6)
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
            if length_inc > 0 :
                candidate_mask = torch.cat([candidate_mask, torch.zeros(length_inc * self.n_offsets, dtype=torch.bool, device='cuda')], dim=0)
                candidate_ds_mask = torch.cat([candidate_ds_mask, torch.zeros(length_inc * self.n_offsets, dtype=torch.bool, device='cuda')], dim=0)
                candidate_extra_mask = torch.cat([candidate_extra_mask, torch.zeros(length_inc, dtype=torch.bool, device='cuda')], dim=0)

            repeated_mask = repeat(level_mask, 'n -> (n k)', k=self.n_offsets)
            candidate_mask = torch.logical_and(candidate_mask, repeated_mask)
            candidate_ds_mask = torch.logical_and(candidate_ds_mask, repeated_mask)
            if ~self.progressive or iteration > self.coarse_intervals[-1]:
                self._extra_level += extra_up * candidate_extra_mask.float()

            all_xyz = self.get_anchor.unsqueeze(dim=1) + self._offset * self.get_scaling[:,:3].unsqueeze(dim=1)

            grid_coords = torch.round((self.get_anchor[level_mask]-self.init_pos)/cur_size).int()
            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round((selected_xyz-self.init_pos)/cur_size).int()
            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)
            if selected_grid_coords_unique.shape[0] > 0 and grid_coords.shape[0] > 0:
                remove_duplicates = self.get_remove_duplicates(grid_coords, selected_grid_coords_unique)
                remove_duplicates = ~remove_duplicates
                candidate_anchor = selected_grid_coords_unique[remove_duplicates]*cur_size+self.init_pos
                new_level = torch.ones(candidate_anchor.shape[0], dtype=torch.int, device='cuda') * cur_level
                candidate_anchor, new_level, _, weed_mask = self.weed_out(candidate_anchor, new_level)
                remove_duplicates_clone = remove_duplicates.clone()
                remove_duplicates[remove_duplicates_clone] = weed_mask
            else:
                candidate_anchor = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates = torch.ones([0], dtype=torch.bool, device='cuda')
                new_level = torch.zeros([0], dtype=torch.int, device='cuda')

            if (~self.progressive or iteration > self.coarse_intervals[-1]) and cur_level < self.levels - 1:
                grid_coords_ds = torch.round((self.get_anchor[level_ds_mask]-self.init_pos)/ds_size).int()
                selected_xyz_ds = all_xyz.view([-1, 3])[candidate_ds_mask]
                selected_grid_coords_ds = torch.round((selected_xyz_ds-self.init_pos)/ds_size).int()
                selected_grid_coords_unique_ds, inverse_indices_ds = torch.unique(selected_grid_coords_ds, return_inverse=True, dim=0)
                if selected_grid_coords_unique_ds.shape[0] > 0 and grid_coords_ds.shape[0] > 0:
                    remove_duplicates_ds = self.get_remove_duplicates(grid_coords_ds, selected_grid_coords_unique_ds)
                    remove_duplicates_ds = ~remove_duplicates_ds
                    candidate_anchor_ds = selected_grid_coords_unique_ds[remove_duplicates_ds]*ds_size+self.init_pos
                    new_level_ds = torch.ones(candidate_anchor_ds.shape[0], dtype=torch.int, device='cuda') * (cur_level + 1)
                    candidate_anchor_ds, new_level_ds, _, weed_ds_mask = self.weed_out(candidate_anchor_ds, new_level_ds)
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
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]
                new_feat_ds = torch.zeros([candidate_anchor_ds.shape[0], self.feat_dim], dtype=torch.float, device='cuda')
                new_feat = torch.cat([new_feat, new_feat_ds], dim=0)

                new_scaling = torch.ones_like(candidate_anchor).repeat([1,2]).float().cuda()*cur_size
                new_scaling_ds = torch.ones_like(candidate_anchor_ds).repeat([1,2]).float().cuda()*ds_size
                new_scaling = torch.cat([new_scaling, new_scaling_ds], dim=0)
                new_scaling = torch.log(new_scaling)

                new_rotation = torch.zeros([candidate_anchor.shape[0], 4], dtype=torch.float, device='cuda')
                new_rotation_ds = torch.zeros([candidate_anchor_ds.shape[0], 4], dtype=torch.float, device='cuda')
                new_rotation = torch.cat([new_rotation, new_rotation_ds], dim=0)
                new_rotation[:,0] = 1.0

                new_opacities = inverse_sigmoid(0.1 * torch.ones((candidate_anchor.shape[0], 1), dtype=torch.float, device="cuda"))
                new_opacities_ds = inverse_sigmoid(0.1 * torch.ones((candidate_anchor_ds.shape[0], 1), dtype=torch.float, device="cuda"))
                new_opacities = torch.cat([new_opacities, new_opacities_ds], dim=0)

                new_offsets = torch.zeros_like(candidate_anchor).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()
                new_offsets_ds = torch.zeros_like(candidate_anchor_ds).unsqueeze(dim=1).repeat([1,self.n_offsets,1]).float().cuda()
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
                }

                temp_anchor_demon = torch.cat([self.anchor_demon, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
                del self.anchor_demon
                self.anchor_demon = temp_anchor_demon

                temp_opacity_accum = torch.cat([self.opacity_accum, torch.zeros([new_opacities.shape[0], 1], device='cuda').float()], dim=0)
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

    def adjust_anchor(self, iteration, check_interval=100, success_threshold=0.8, grad_threshold=0.0002, update_ratio=0.5, extra_ratio=4.0, extra_up=0.25, min_opacity=0.005):
        # # adding anchors
        grads = self.offset_gradient_accum / self.offset_denom
        grads[grads.isnan()] = 0.0
        grads_norm = torch.norm(grads, dim=-1)
        offset_mask = (self.offset_denom > check_interval*success_threshold*0.5).squeeze(dim=1)

        self.anchor_growing(iteration, grads_norm, grad_threshold, update_ratio, extra_ratio, extra_up, offset_mask)

        # update offset_denom
        self.offset_denom[offset_mask] = 0
        padding_offset_demon = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_denom.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_denom.device)
        self.offset_denom = torch.cat([self.offset_denom, padding_offset_demon], dim=0)

        self.offset_gradient_accum[offset_mask] = 0
        padding_offset_gradient_accum = torch.zeros([self.get_anchor.shape[0]*self.n_offsets - self.offset_gradient_accum.shape[0], 1],
                                           dtype=torch.int32,
                                           device=self.offset_gradient_accum.device)
        self.offset_gradient_accum = torch.cat([self.offset_gradient_accum, padding_offset_gradient_accum], dim=0)

        # # prune anchors
        prune_mask = (self.opacity_accum < min_opacity*self.anchor_demon).squeeze(dim=1)
        anchors_mask = (self.anchor_demon > check_interval*success_threshold).squeeze(dim=1)
        prune_mask = torch.logical_and(prune_mask, anchors_mask)

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
        if anchors_mask.sum()>0:
            self.opacity_accum[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()
            self.anchor_demon[anchors_mask] = torch.zeros([anchors_mask.sum(), 1], device='cuda').float()

        temp_opacity_accum = self.opacity_accum[~prune_mask]
        del self.opacity_accum
        self.opacity_accum = temp_opacity_accum

        temp_anchor_demon = self.anchor_demon[~prune_mask]
        del self.anchor_demon
        self.anchor_demon = temp_anchor_demon

        if prune_mask.shape[0]>0:
            self.prune_anchor(prune_mask)

    def save_mlp_checkpoints(self, path, mode = 'split', model_name = ''):#split or unite
        mkdir_p(os.path.dirname(path))
        if mode == 'split':
            self.eval()
            opacity_mlp = torch.jit.trace(self.mlp_opacity, (torch.rand(1, self.feat_dim+self.view_dim+self.opacity_dist_dim+self.level_dim).cuda()))
            opacity_mlp.save(os.path.join(path, f'opacity_mlp_{model_name}.pt'))
            cov_mlp = torch.jit.trace(self.mlp_cov, (torch.rand(1, self.feat_dim+self.view_dim+self.cov_dist_dim+self.level_dim).cuda()))
            cov_mlp.save(os.path.join(path, f'cov_mlp_{model_name}.pt'))
            color_mlp = torch.jit.trace(self.mlp_color, (torch.rand(1, self.feat_dim+self.view_dim+self.color_dist_dim+self.appearance_dim+self.level_dim).cuda()))
            color_mlp.save(os.path.join(path, f'color_mlp_{model_name}.pt'))
            if self.use_feat_bank:
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, (torch.rand(1, 3+self.level_dim).cuda()))
                feature_bank_mlp.save(os.path.join(path, f'feature_bank_mlp_{model_name}.pt'))
            if self.appearance_dim > 0:
                emd = torch.jit.trace(self.embedding_appearance, (torch.zeros((1,), dtype=torch.long).cuda()))
                emd.save(os.path.join(path, f'embedding_appearance_{model_name}.pt'))
            self.train()
        elif mode == 'unite':
            param_dict = {}
            param_dict['opacity_mlp'] = self.mlp_opacity.state_dict()
            param_dict['cov_mlp'] = self.mlp_cov.state_dict()
            param_dict['color_mlp'] = self.mlp_color.state_dict()
            if self.use_feat_bank:
                param_dict['feature_bank_mlp'] = self.mlp_feature_bank.state_dict()
            if self.appearance_dim > 0:
                param_dict['appearance'] = self.embedding_appearance.state_dict()
            torch.save(param_dict, os.path.join(path, f'checkpoints_{model_name}.pth'))
        else:
            raise NotImplementedError

    def make_mlp_checkpoints(self, path, mode='split'):
        mkdir_p(os.path.dirname(path))
        ptdata_list = []  #  PlyElement

        if mode == 'split':
            self.eval()

            #  MLP
            opacity_input = torch.rand(1, self.feat_dim + self.view_dim + self.opacity_dist_dim + self.level_dim).cuda()
            opacity_mlp = torch.jit.trace(self.mlp_opacity, opacity_input)
            opacity_output = opacity_mlp(opacity_input)
            ptdata_list.append(self.convert_to_ply_element(opacity_output, 'opacity'))

            cov_input = torch.rand(1, self.feat_dim + self.view_dim + self.cov_dist_dim + self.level_dim).cuda()
            cov_mlp = torch.jit.trace(self.mlp_cov, cov_input)
            cov_output = cov_mlp(cov_input)
            ptdata_list.append(self.convert_to_ply_element(cov_output, 'cov'))

            color_input = torch.rand(1,
                                     self.feat_dim + self.view_dim + self.color_dist_dim + self.appearance_dim + self.level_dim).cuda()
            color_mlp = torch.jit.trace(self.mlp_color, color_input)
            color_output = color_mlp(color_input)
            ptdata_list.append(self.convert_to_ply_element(color_output, 'color'))

            if self.use_feat_bank:
                feature_bank_input = torch.rand(1, 3 + self.level_dim).cuda()
                feature_bank_mlp = torch.jit.trace(self.mlp_feature_bank, feature_bank_input)
                feature_bank_output = feature_bank_mlp(feature_bank_input)
                ptdata_list.append(self.convert_to_ply_element(feature_bank_output, 'feature_bank'))

            if self.appearance_dim > 0:
                emd_input = torch.zeros((1,), dtype=torch.long).cuda()
                emd = torch.jit.trace(self.embedding_appearance, emd_input)
                emd_output = emd(emd_input)
                ptdata_list.append(self.convert_to_ply_element(emd_output, 'appearance'))

            self.train()

        elif mode == 'unite':
            param_dict = {
                'opacity_mlp': self.mlp_opacity.state_dict(),
                'cov_mlp': self.mlp_cov.state_dict(),
                'color_mlp': self.mlp_color.state_dict()
            }

            if self.use_feat_bank:
                param_dict['feature_bank_mlp'] = self.mlp_feature_bank.state_dict()

            if self.appearance_dim > 0:
                param_dict['appearance'] = self.embedding_appearance.state_dict()

            # , checkpoints
            if str(path).endswith(('.pth', '.pt')):
                save_path = path
            else:
                save_path = os.path.join(path, 'checkpoints.pth')
            torch.save(param_dict, save_path)
        else:
            raise NotImplementedError

        return ptdata_list  #  PlyElement

    def convert_to_ply_element(self, mlp_output, element_name):
        #  mlp_output  torch.Tensor , NumPy
        #  mlp_output
        data = mlp_output.cpu().detach().numpy()  #  GPU  CPU, NumPy

        #  x, y, z
        vertex = np.array(data, dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])

        #  PlyElement
        return PlyElement.describe(vertex, element_name)

    def load_mlp_checkpoints(self, path, mode = 'split', model_name = ''):#split or unite
        print("loading model from exists{}".format(path))
        if mode == 'split':
            self.mlp_opacity = torch.jit.load(os.path.join(path, f'opacity_mlp_{model_name}.pt')).cuda()
            self.mlp_cov = torch.jit.load(os.path.join(path, f'cov_mlp_{model_name}.pt')).cuda()
            self.mlp_color = torch.jit.load(os.path.join(path, f'color_mlp_{model_name}.pt')).cuda()
            if self.use_feat_bank:
                self.mlp_feature_bank = torch.jit.load(os.path.join(path, f'feature_bank_mlp_{model_name}.pt')).cuda()
            if self.appearance_dim > 0:
                self.embedding_appearance = torch.jit.load(os.path.join(path, f'embedding_appearance_{model_name}.pt')).cuda()
        elif mode == 'unite':
            checkpoint = torch.load(os.path.join(path, f'checkpoints_{model_name}.pth'))
            self.mlp_opacity.load_state_dict(checkpoint['opacity_mlp'])
            self.mlp_cov.load_state_dict(checkpoint['cov_mlp'])
            self.mlp_color.load_state_dict(checkpoint['color_mlp'])
            if self.use_feat_bank:
                self.mlp_feature_bank.load_state_dict(checkpoint['feature_bank_mlp'])
            if self.appearance_dim > 0:
                self.embedding_appearance.load_state_dict(checkpoint['appearance'])
        else:
            raise NotImplementedError

    @property
    def get_normal(self):
        return self.normal_activation(self._normal)
    @property
    def get_semantic(self):
        if self.semantic_mode == 'logits':
            return self._semantic
        elif self.semantic_mode == 'probabilities':
            return torch.nn.functional.softmax(self._semantic, dim=1)

    def parse_camera(self, camera):
        pass
