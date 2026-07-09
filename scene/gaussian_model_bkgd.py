#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
from datetime import timedelta
import time
from typing import Optional
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
from utils.general_utils import quaternion_to_matrix, build_scaling_rotation, strip_symmetric, quaternion_raw_multiply, startswith_any, matrix_to_quaternion
from bidict import bidict
from utils.camera_utils import Camera

class GaussianModelBkgd(GaussianModel):

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
        super().__init__(sh_degree=sh_degree, args=args, model_name=model_name)

        # ===== Hash encoding: (waymo_default.py  *_bkgd )=====
        #  args.hash_levels_bkgd / hash_base_resolution_bkgd  hash_* ,
        # "", anchors  -> render .
        if hasattr(args, "hash_levels_bkgd"):
            self.hash_levels = getattr(args, "hash_levels_bkgd")
        if hasattr(args, "hash_base_resolution_bkgd"):
            self.hash_base_resolution = getattr(args, "hash_base_resolution_bkgd")
        if hasattr(args, "hash_finest_resolution_bkgd"):
            self.hash_finest_resolution = getattr(args, "hash_finest_resolution_bkgd")
        if hasattr(args, "hash_log2_size_bkgd"):
            self.hash_log2_size = getattr(args, "hash_log2_size_bkgd")
        self.anchor_generation_levels = getattr(args, "anchor_generation_levels", None)
        if hasattr(args, "use_hash_feat_single_level_bkgd"):
            self.use_hash_feat_single_level = bool(getattr(args, "use_hash_feat_single_level_bkgd"))
        #  hash table
        self.hash_encoding = None
        self.dynamic_voxel_sizes = {}
        self.region_bounds = {}
        self.near_is_low = True
        self.bounds = None
        self.main_direction = None
        self.init_pos = None
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
        self.visible_mask = None
        self.rendered_opacity = None
        self.rendered_anchor_mask = None

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

        self.background_mask = None

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
        num_level = self.levels - 1 - self.init_level
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
        if hasattr(self, 'bounds') and self.bounds is not None:
            self.base_bounds = self.bounds.copy() if isinstance(self.bounds, list) else list(self.bounds)
        else:
            self.base_bounds = None

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

    def determine_near_far_direction(self, proj_coords, first_cam_center=None):
        """


        :
        1. :,,
        2. :,,

        :
        -
        - ,

        Args:
            proj_coords: (N,) torch.Tensor,
            first_cam_center: (3,) torch.Tensor,
        Returns:
            near_is_low: bool, True ,False
        """

        if first_cam_center is not None:

            min_proj = torch.min(proj_coords)
            max_proj = torch.max(proj_coords)
            mid_proj = (min_proj + max_proj) / 2


            cam_proj = torch.matmul(first_cam_center - self.init_pos, self.main_direction)





            if cam_proj < mid_proj:

                near_is_low = True
                print(f"[Near-Far] Camera at small side (cam_proj={cam_proj:.2f} < mid={mid_proj:.2f}), near_is_low=True")
            else:

                near_is_low = False
                print(f"[Near-Far] Camera at large side (cam_proj={cam_proj:.2f} > mid={mid_proj:.2f}), near_is_low=False")



            mask_low = proj_coords < mid_proj
            mask_high = proj_coords > mid_proj
            density_low = mask_low.sum().item() / (mask_low.sum().item() + 1e-6)
            density_high = mask_high.sum().item() / (mask_high.sum().item() + 1e-6)

            print(f"[Near-Far] Density: low_side={density_low:.3f}, high_side={density_high:.3f}")
            print(f"[Near-Far] Range: min={min_proj:.2f}, max={max_proj:.2f}, cam={cam_proj:.2f}")

            return near_is_low


        from sklearn.mixture import GaussianMixture

        proj_coords_np = proj_coords.cpu().numpy().reshape(-1, 1)


        gmm = GaussianMixture(n_components=3, random_state=42)
        gmm.fit(proj_coords_np)
        means = torch.tensor(gmm.means_).flatten().sort()[0]

        #  GMM
        labels = torch.tensor(gmm.predict(proj_coords_np))


        densities = torch.bincount(labels) / len(labels)
        variances = [proj_coords[labels == i].var().item() for i in range(3)]


        near_idx = densities.argmax().item()
        far_idx = variances.index(max(variances))

        near_is_low = means[near_idx] < means[far_idx]
        return near_is_low

    def adaptive_layer_segmentation(self, proj_coords):
        """
        (GMM),,

        :
        1.  GMM  3 (,,)
        2. :()
        3. : self.near_is_low ()
        4. :

        Args:
            proj_coords: (N,) torch.Tensor,
        Returns:
            bounds: list,  [near_bound, far_bound]
        """
        proj_coords_np = proj_coords.cpu().numpy().reshape(-1, 1)

        #  GMM
        from sklearn.mixture import GaussianMixture
        gmm = GaussianMixture(n_components=3, covariance_type='full', random_state=42)
        gmm.fit(proj_coords_np)


        means_original = torch.tensor(gmm.means_).flatten()
        means_sorted, sort_indices = torch.sort(means_original)

        # ( GMM )
        labels = torch.tensor(gmm.predict(proj_coords_np))
        densities = torch.bincount(labels, minlength=3).float() / len(labels)
        variances = [proj_coords[labels == i].var().item() if (labels == i).sum() > 0 else 0.0 for i in range(3)]

        # ()-  GMM
        far_idx_original = variances.index(max(variances))


        far_idx_sorted = (sort_indices == far_idx_original).nonzero(as_tuple=True)[0].item()


        # means_sorted: [mean_0, mean_1, mean_2] ()
        # far_idx_sorted:  (0, 1,  2)

        if self.near_is_low:

            near_idx = 0
            if far_idx_sorted == 0:

                print(f"[Warning] Farsmall side,large side")
                far_idx_sorted = 2
                mid_idx = 1
            elif far_idx_sorted == 1:

                mid_idx = 2
            else:  # far_idx_sorted == 2

                mid_idx = 1
        else:

            near_idx = 2
            if far_idx_sorted == 2:

                print(f"[Warning] Farlarge side,small side")
                far_idx_sorted = 0
                mid_idx = 1
            elif far_idx_sorted == 1:

                mid_idx = 0
            else:  # far_idx_sorted == 0

                mid_idx = 1


        # near_bound:
        # far_bound:
        near_mean = means_sorted[near_idx]
        mid_mean = means_sorted[mid_idx]
        far_mean = means_sorted[far_idx_sorted]

        near_bound = (near_mean + mid_mean) / 2
        far_bound = (mid_mean + far_mean) / 2

        print(f"[GMM Segmentation] near={near_mean:.2f}, mid={mid_mean:.2f}, far={far_mean:.2f}")
        print(f"[GMM Segmentation] near_bound={near_bound:.2f}, far_bound={far_bound:.2f}")
        print(f"[GMM Segmentation] Variances: {variances}, far_idx_original={far_idx_original}, far_idx_sorted={far_idx_sorted}")

        return [near_bound.item(), far_bound.item()]

    def compute_adaptive_voxel_sizes(self, data, proj_coords, base_voxel_size):
        """

        Args:
            data: (N,3) torch.Tensor,
            proj_coords: (N,) torch.Tensor,
            base_voxel_size: float,
        Returns:
            dynamic_voxel_sizes: dict, {'near': xxx, 'mid': xxx, 'far': xxx}
        """
        #  proj_coords
        proj_coords = proj_coords.view(-1)


        min_proj = proj_coords.min()
        max_proj = proj_coords.max()
        if max_proj > min_proj:
            normalized_proj = (proj_coords - min_proj) / (max_proj - min_proj) * 99
        else:
            normalized_proj = torch.zeros_like(proj_coords)


        bin_indices = normalized_proj.long()


        density = 1.0 / (torch.bincount(bin_indices, minlength=100).float() + 1e-5)


        density = density / density.max()
        mean_density = density.mean().item()
        min_density = density.min().item()


        return {
            "near": base_voxel_size / 10,
            "mid": base_voxel_size / (4 + 2 * mean_density),
            "far": base_voxel_size / (1 + 4 * min_density)
        }


    def lod_tree_sample(self, data, init_pos, first_cam_center=None):

        self.init_pos = init_pos
        torch.cuda.synchronize(); t0 = time.time()
        self.positions = torch.empty(0, 3).float().cuda()
        self._level = torch.empty(0).int().cuda()


        centered_data = data - data.mean(dim=0)
        # PCA
        # U, S, V = torch.pca_lowrank(centered_data, q=1)
        U, S, V = torch.pca_lowrank(centered_data, q=min(3, centered_data.shape[1]))
        main_direction = V[:, 0]
        self.main_direction = main_direction


        proj_coords = torch.matmul(data - init_pos, main_direction)

        self.near_is_low = self.determine_near_far_direction(proj_coords, first_cam_center)
        near_is_low = self.near_is_low

        self.bounds = self.adaptive_layer_segmentation(proj_coords)
        bounds = self.bounds

        self.dynamic_voxel_sizes = self.compute_adaptive_voxel_sizes(data, proj_coords, self.voxel_size)

        for cur_level in range(self.levels):

            for region in ["near", "mid", "far"]:
                cur_size = self.dynamic_voxel_sizes[region] / (float(self.fork) ** cur_level) / 1
                if region == "near":

                    if near_is_low:
                        mask = proj_coords <= bounds[0]
                    else:
                        mask = proj_coords >= bounds[0]
                elif region == "mid":

                    if near_is_low:
                        mask = (proj_coords > bounds[0]) & (proj_coords < bounds[1])
                    else:
                        mask = (proj_coords > bounds[1]) & (proj_coords < bounds[0])
                else:

                    if near_is_low:
                        mask = proj_coords >= bounds[1]
                    else:
                        mask = proj_coords <= bounds[1]
                if not mask.any():
                    continue
                region_data = data[mask]
                new_positions = torch.round((region_data - init_pos) / cur_size) * cur_size + init_pos
                new_positions = torch.unique(new_positions, dim=0)

                new_level = torch.full((new_positions.shape[0],), cur_level, dtype=torch.int, device="cuda")
                self.positions = torch.concat((self.positions, new_positions), dim=0)
                self._level = torch.concat((self._level, new_level), dim=0)
                import open3d as o3d
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(data[mask].cpu().numpy().astype(np.float64))  #  float64

                o3d.io.write_point_cloud(f"outputs/{region}_level{cur_level}.ply", pcd)
        torch.cuda.synchronize(); t1 = time.time()
        print(f"[{getattr(self, 'model_name', 'background')}] Building LOD tree time: {int(t1 - t0) // 60} min {int(t1 - t0) % 60} sec")

    def create_from_pcd(self, points: BasicPointCloud, spatial_lr_scale: float, first_cam_center=None, logger=None):
        self.spatial_lr_scale = spatial_lr_scale
        xyz_min = torch.min(points, dim=0).values * self.extend
        xyz_max = torch.max(points, dim=0).values * self.extend
        box_d = xyz_max - xyz_min
        if self.base_layer < 0:
            default_voxel_size = 0.02
            scalar_box = torch.max(box_d)
            self.base_layer = int(torch.round(torch.log2(scalar_box/default_voxel_size)).item())-(self.levels//2)+1
        print(f"base_layer: {self.base_layer}")
        print(f"box_d: {box_d}")
        print(f"fork: {self.fork}")
        self.voxel_size = box_d/(float(self.fork) ** self.base_layer)
        self.init_pos = xyz_min.clone().float().cuda()
        used_hash = bool(getattr(self, 'use_hash_encoding', False))
        #  hash_sample  voxel_size( vector  scalar)
        pre_voxel_size = self.voxel_size.clone() if isinstance(self.voxel_size, torch.Tensor) else self.voxel_size
        #  hash_sample ( hash  LOD tree )
        try:
            self.pre_hash_base_voxel = pre_voxel_size.max().detach().cpu() if isinstance(pre_voxel_size, torch.Tensor) else float(pre_voxel_size)
        except Exception:
            self.pre_hash_base_voxel = float(pre_voxel_size) if not isinstance(pre_voxel_size, torch.Tensor) else float(pre_voxel_size.max().item())
        print(f"[{getattr(self, 'model_name', 'background')}] init sampling: use_hash_encoding={used_hash}")
        if used_hash:
            #  hash (AABB  xyz_min/xyz_max)
            #  pre-hash , hash_sample(hash_sample  self.voxel_size)
            #  base_voxel_size  total_levels, LOD tree
            default_voxel_size = 0.02
            scalar_box = torch.max(box_d)
            base_layer = int(torch.round(torch.log2(scalar_box/default_voxel_size)).item())-(self.levels//2)+1
            base_voxel_size = box_d / (float(self.fork) ** base_layer)
            anchor_levels = getattr(self, "anchor_generation_levels", None)
            if anchor_levels is not None and int(anchor_levels) > 0:
                total_levels = int(anchor_levels)
            else:
                total_levels = base_layer + 1
            print(f"[create_from_pcd-hash] box_d={box_d}, base_layer={base_layer}, base_voxel_size={base_voxel_size}, total_levels={total_levels}, anchor_generation_levels={anchor_levels}")
            self.hash_sample(points, xyz_min, xyz_max, base_voxel_size=base_voxel_size.max().item() if isinstance(base_voxel_size, torch.Tensor) else float(base_voxel_size), levels=total_levels)
            centered_data = points - points.mean(dim=0)
            # PCA ()
            _, _, V = torch.pca_lowrank(centered_data, q=min(3, centered_data.shape[1]))
            self.main_direction = V[:, 0]
            if self.main_direction is None:
                raise RuntimeError(f"[{getattr(self, 'model_name', 'background')}] main_direction is None after PCA (unexpected).")

            proj_coords = torch.matmul(points - self.init_pos, self.main_direction)

            self.near_is_low = self.determine_near_far_direction(proj_coords, first_cam_center)
            self.bounds = self.adaptive_layer_segmentation(proj_coords)
            if self.bounds is None or len(self.bounds) < 2:
                raise RuntimeError(f"[{getattr(self, 'model_name', 'background')}] bounds invalid after adaptive_layer_segmentation: {self.bounds}")

            # (// voxel offset)
            self.dynamic_voxel_sizes = self.compute_adaptive_voxel_sizes(points, proj_coords, self.voxel_size)
            #  base_bounds  RL/
            self.base_bounds = self.bounds.copy() if isinstance(self.bounds, list) else list(self.bounds)


            import open3d as o3d
            import os
            os.makedirs("outputs", exist_ok=True)

            bounds = self.bounds
            near_is_low = self.near_is_low

            for region in ["near", "mid", "far"]:
                if region == "near":

                    if near_is_low:
                        mask = proj_coords <= bounds[0]
                    else:
                        mask = proj_coords >= bounds[0]
                elif region == "mid":

                    if near_is_low:
                        mask = (proj_coords > bounds[0]) & (proj_coords < bounds[1])
                    else:
                        mask = (proj_coords > bounds[1]) & (proj_coords < bounds[0])
                else:  # region == "far"

                    if near_is_low:
                        mask = proj_coords >= bounds[1]
                    else:
                        mask = proj_coords <= bounds[1]

                if mask.any():
                    region_data = points[mask]
                    pcd = o3d.geometry.PointCloud()
                    pcd.points = o3d.utility.Vector3dVector(region_data.cpu().numpy().astype(np.float64))
                    # (hash  level, level)
                    output_path = f"outputs/{getattr(self, 'model_name', 'background')}_hash_{region}.ply"
                    o3d.io.write_point_cloud(output_path, pcd)
                    print(f"[{getattr(self, 'model_name', 'background')}] Saved {region} region point cloud: {output_path} ({mask.sum().item()} points)")
                else:
                    print(f"[{getattr(self, 'model_name', 'background')}] Warning: {region} region has no points, skipping save")
        else:
            self.lod_tree_sample(points, self.init_pos, first_cam_center)

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
            print(f'Background LOD Levels (final): {self.levels}')
            print(f'Final Voxel Size (post-hash): {self.voxel_size}')
        else:
            print(f'Base Layer of Tree: {self.base_layer}')
            print(f'Background LOD Levels: {self.levels}')
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
            logger.info(f'Background LOD Levels (final): {self.levels}')
            logger.info(f'Final Voxel Size (post-hash): {self.voxel_size}')
        else:
            logger.info(f'Base Layer of Tree: {self.base_layer}')
            logger.info(f'Background LOD Levels: {self.levels}')
            logger.info(f'Max Voxel Size: {self.voxel_size}')
            logger.info(f'Min Voxel Size: {self.voxel_size/(2.0 ** (self.levels - 1))}')

        logger.info(f'Visible Threshold: {self.visible_threshold}')
        logger.info(f'Appearance Embedding Dimension: {self.appearance_dim}')
        logger.info(f'Initial Levels: {self.init_level}')
        logger.info(f'Initial Voxel Number: {self.positions.shape[0]}')

        offsets = torch.zeros((self.positions.shape[0], self.n_offsets, 3)).float().cuda()
        # LOD tree:hash,0
        # hash_lod0,hash
        anchors_feat = torch.zeros((self.positions.shape[0], self.feat_dim)).float().cuda()
        #  anchor , distCUDA2  CUDA (invalid configuration)
        #  batch ,.
        def _safe_dist2(positions, batch_size=2000000):
            device = positions.device
            parts = []
            N = positions.shape[0]
            for i in range(0, N, batch_size):
                chunk = positions[i:i+batch_size]
                d = distCUDA2(chunk).float().to(device)
                parts.append(d)
            if len(parts) == 0:
                return torch.empty(0, device=device)
            return torch.cat(parts, dim=0)

        dist2 = torch.clamp_min(_safe_dist2(self.positions), 0.0000001)
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
        if self.use_hash_encoding and int(getattr(self, "levels", 0)) > 1:
            try:
                base_res = float(getattr(self, "hash_base_resolution"))
                finest_res = float(getattr(self, "hash_finest_resolution"))
                lod_base = (finest_res / base_res) ** (1.0 / float(self.levels - 1))
            except Exception:
                lod_base = float(self.fork)
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

    def compute_region_labels(self, anchors):
        """  """
        anchor_proj = torch.matmul(anchors - self.init_pos, self.main_direction)

        if self.near_is_low:
            near_mask = anchor_proj <= self.bounds[0]
            far_mask = anchor_proj > self.bounds[1]
        else:
            near_mask = anchor_proj >= self.bounds[0]
            far_mask = anchor_proj < self.bounds[1]

        mid_mask = ~(near_mask | far_mask)

        self.region_labels = torch.full_like(anchor_proj, 2, dtype=torch.long)
        self.region_labels[near_mask] = 0
        self.region_labels[mid_mask] = 1
        return self.region_labels

    def get_dynamic_voxel_size(self):
        self.dynamic_voxel_sizes = {
                "near": self.voxel_size / 10,
                "mid": self.voxel_size / 5,
                "far": self.voxel_size / 1
            }

    def set_anchor_mask(self, cam_center, iteration, resolution_scale):
        # hash encodingLOD,anchor
        if self.use_hash_encoding and self.hash_disable_lod:
            self._anchor_mask = torch.ones(self._anchor.shape[0], dtype=torch.bool, device=self._anchor.device)
            return

        # LOD
        anchor_pos = self._anchor.clone()
        self.get_dynamic_voxel_size()
        labels = self.compute_region_labels(anchor_pos)
        # hash  level  per_level_scale( fork=2), LOD/offset
        if self.use_hash_encoding and getattr(self, "hash_encoding", None) is not None:
            lod_base = float(self.hash_encoding.per_level_scale)
        else:
            lod_base = float(self.fork)
        lod_base_t = torch.tensor(lod_base, device=self._level.device, dtype=torch.float32)

        m0 = labels == 0
        if torch.any(m0):
            scale0 = torch.pow(lod_base_t, self._level[m0].float())
            anchor_pos[m0] += self.dynamic_voxel_sizes["near"] / scale0
        m1 = labels == 1
        if torch.any(m1):
            scale1 = torch.pow(lod_base_t, self._level[m1].float())
            anchor_pos[m1] += self.dynamic_voxel_sizes["mid"] / scale1
        m2 = labels == 2
        if torch.any(m2):
            scale2 = torch.pow(lod_base_t, self._level[m2].float())
            anchor_pos[m2] += self.dynamic_voxel_sizes["far"] / scale2
        dist = torch.sqrt(torch.sum((anchor_pos - cam_center)**2, dim=1)) * resolution_scale
        #  level
        pred_level = torch.log(self.standard_dist/dist) / math.log(lod_base) + self._extra_level

        dist_world = dist / resolution_scale if resolution_scale != 0 else dist
        level_bias = torch.zeros_like(pred_level)
        level_bias = level_bias + (dist_world > self.lod_depth_near).float() * self.lod_depth_bias_mid
        level_bias = level_bias + (dist_world > self.lod_depth_mid).float() * (self.lod_depth_bias_far - self.lod_depth_bias_mid)
        pred_level = torch.clamp(pred_level - level_bias, min=0.0)

        is_training = self.get_color_mlp.training
        if self.progressive and is_training:
            coarse_index = np.searchsorted(self.coarse_intervals, iteration) + 1 + self.init_level
        else:
            coarse_index = self.levels

        # hash encodingLOD,max_level
        max_level = coarse_index - 1
        if self.use_hash_encoding and getattr(self, 'hash_levels', None) is not None and int(getattr(self, 'hash_levels', 0)) > 0:
            max_level = min(max_level, int(self.hash_levels) - 1)
        if iteration is not None and iteration < self.lod_warmup_iters:
            max_level = min(max_level, self.lod_warmup_max_level)

        int_level = self.map_to_int_level(pred_level, max_level)
        self._anchor_mask = (self._level.squeeze(dim=1) <= int_level)

    def set_anchor_mask_perlevel(self, cam_center, resolution_scale, cur_level):
        anchor_pos = self._anchor.clone()
        labels = self.compute_region_labels(anchor_pos)
        if self.use_hash_encoding and getattr(self, "hash_encoding", None) is not None:
            lod_base = float(self.hash_encoding.per_level_scale)
        else:
            lod_base = float(self.fork)
        lod_base_t = torch.tensor(lod_base, device=self._level.device, dtype=torch.float32)

        m0 = labels == 0
        if torch.any(m0):
            scale0 = torch.pow(lod_base_t, self._level[m0].float())
            anchor_pos[m0] += self.dynamic_voxel_sizes["near"] / scale0
        m1 = labels == 1
        if torch.any(m1):
            scale1 = torch.pow(lod_base_t, self._level[m1].float())
            anchor_pos[m1] += self.dynamic_voxel_sizes["mid"] / scale1
        m2 = labels == 2
        if torch.any(m2):
            scale2 = torch.pow(lod_base_t, self._level[m2].float())
            anchor_pos[m2] += self.dynamic_voxel_sizes["far"] / scale2
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
        l.append('info')
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
        for i in range(self.main_direction.shape[0]):
            l.append('main_direction_{}'.format(i))
        for i in range(self.init_pos.shape[0]):
            l.append('init_pos_{}'.format(i))
        bounds_array = np.array(self.bounds)
        for i in range(bounds_array.shape[0]):
            l.append('bounds_{}'.format(i))
        return l

    def save_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        levels = self._level.detach().cpu().numpy()
        extra_levels = self._extra_level.unsqueeze(dim=1).detach().cpu().numpy()
        infos = np.zeros_like(levels, dtype=np.float32)
        infos[0, 0] = self.voxel_size
        infos[1, 0] = self.standard_dist

        anchor_feats = self._anchor_feat.detach().cpu().numpy()
        offsets = self._offset.detach().transpose(1, 2).flatten(start_dim=1).contiguous().cpu().numpy()
        opacities = self._opacity.detach().cpu().numpy()
        scales = self._scaling.detach().cpu().numpy()
        rots = self._rotation.detach().cpu().numpy()
        semantic = self._semantic.detach().cpu().numpy()

        main_direction = np.tile(self.main_direction.detach().cpu().numpy(), (anchor.shape[0], 1))
        init_pos = np.tile(self.init_pos.detach().cpu().numpy(), (anchor.shape[0], 1))
        bounds = np.array(self.bounds).flatten()
        bounds = np.tile(bounds, (anchor.shape[0], 1))

        dtype_full = [(attribute, 'f4') for attribute in self.construct_list_of_attributes()]

        elements = np.empty(anchor.shape[0], dtype=dtype_full)
        attributes = np.concatenate((anchor, levels, extra_levels, infos, offsets, anchor_feats, opacities, scales, rots, semantic, main_direction, init_pos, bounds), axis=1)
        elements[:] = list(map(tuple, attributes))
        el = PlyElement.describe(elements, 'vertex')
        PlyData([el]).write(path)

    def make_ply(self, path):
        mkdir_p(os.path.dirname(path))

        anchor = self._anchor.detach().cpu().numpy()
        levels = self._level.detach().cpu().numpy()
        extra_levels = self._extra_level.unsqueeze(dim=1).detach().cpu().numpy()
        infos = np.zeros_like(levels, dtype=np.float32)
        infos[0, 0] = self.voxel_size
        infos[1, 0] = self.standard_dist

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

        return elements

    def plot_levels(self):
        for level in range(self.levels):
            level_mask = (self._level == level).squeeze(dim=1)
            print(f'Level {level}: {torch.sum(level_mask).item()}, Ratio: {torch.sum(level_mask).item()/self._level.shape[0]}')

    def load_ply_sparse_gaussian(self, path=None, input_ply=None):
        if path is None:
            plydata = input_ply
        else:
            plydata = PlyData.read(path)
            plydata = plydata.elements[0]

        anchor = np.stack((np.asarray(plydata["x"]),
                        np.asarray(plydata["y"]),
                        np.asarray(plydata["z"])),  axis=1).astype(np.float32)
        levels = np.asarray(plydata["level"])[... ,np.newaxis].astype(np.int)
        extra_levels = np.asarray(plydata["extra_level"])[... ,np.newaxis].astype(np.float32)
        self.voxel_size = torch.tensor(plydata["info"][0]).float()
        self.standard_dist = torch.tensor(plydata["info"][1]).float()

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

        main_direction = np.stack((np.asarray(plydata.elements[0]["main_direction_0"]),
                                   np.asarray(plydata.elements[0]["main_direction_1"]),
                                   np.asarray(plydata.elements[0]["main_direction_2"])), axis=1).astype(np.float32)
        init_pos = np.stack((np.asarray(plydata.elements[0]["init_pos_0"]),
                             np.asarray(plydata.elements[0]["init_pos_1"]),
                             np.asarray(plydata.elements[0]["init_pos_2"])), axis=1).astype(np.float32)
        bounds = np.stack((np.asarray(plydata.elements[0]["bounds_0"]),
                           np.asarray(plydata.elements[0]["bounds_1"])), axis=1).astype(np.float32)

        #  main_direction  init_pos
        if main_direction.shape[0] > 1 and np.all(main_direction == main_direction[0]):
            main_direction = main_direction[0]
        if init_pos.shape[0] > 1 and np.all(init_pos == init_pos[0]):
            init_pos = init_pos[0]

        self.main_direction = torch.tensor(main_direction, dtype=torch.float, device="cuda")
        self.init_pos = torch.tensor(init_pos, dtype=torch.float, device="cuda")
        self.bounds = bounds[0].tolist()

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
    def training_statis(self, viewspace_point_tensor_grad, opacity, visibility_mask, opacity_mask, anchor_mask):

        # update_filter->visibility_filter->visibility_mask
        # offset_selection_mask->offset_selection_mask->opacity_mask
        # anchor_visible_mask->voxel_visible_mask->anchor_mask

        # update opacity stats
        temp_opacity = opacity.clone().view(-1).detach()
        temp_opacity[temp_opacity<0] = 0

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
        # combine_mask[temp_mask]visible_mask,combine_mask,false

        # combined_mask[temp_mask] = visibility_mask
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
            anchor_pos = self._anchor.clone()
            labels = self.compute_region_labels(anchor_pos)
            update_value = self.fork ** update_ratio
            level_mask = (self.get_level == cur_level).squeeze(dim=1)
            level_ds_mask = (self.get_level == cur_level + 1).squeeze(dim=1)
            if torch.sum(level_mask) == 0:
                continue
            cur_size = self.voxel_size / (float(self.fork) ** cur_level)
            ds_size = cur_size / self.fork
            cur_size_scalar = float(cur_size.detach().max().item()) if torch.is_tensor(cur_size) else float(cur_size)
            ds_size_scalar = float(ds_size.detach().max().item()) if torch.is_tensor(ds_size) else float(ds_size)
            adjusted_cur_size = torch.full(anchor_pos.shape, cur_size_scalar, dtype=torch.float, device=anchor_pos.device)
            adjusted_ds_size = torch.full(anchor_pos.shape, ds_size_scalar, dtype=torch.float, device=anchor_pos.device)
            adjusted_cur_size[labels == 0] = self.dynamic_voxel_sizes["near"] / (float(self.fork) ** cur_level)
            adjusted_cur_size[labels == 1] = self.dynamic_voxel_sizes["mid"] / (float(self.fork) ** cur_level)
            adjusted_cur_size[labels == 2] = self.dynamic_voxel_sizes["far"] / (float(self.fork) ** cur_level)
            adjusted_ds_size = adjusted_cur_size / (float(self.fork))
            # update threshold
            cur_threshold = threshold * (update_value ** cur_level)
            ds_threshold = cur_threshold * update_value
            extra_threshold = cur_threshold * extra_ratio
            # mask from grad threshold
            candidate_mask = (grads >= cur_threshold) & (grads < ds_threshold)
            candidate_ds_mask = (grads >= ds_threshold)
            candidate_extra_mask = (anchor_grads >= extra_threshold)

            length_inc = self.get_anchor.shape[0] - init_length
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

            grid_coords = torch.round((self.get_anchor[level_mask]-self.init_pos)/adjusted_cur_size[level_mask]).int()
            candidate_indices = torch.nonzero(candidate_mask, as_tuple=False).squeeze(1)
            selected_anchor_indices = candidate_indices // self.n_offsets
            selected_xyz_size = adjusted_cur_size[selected_anchor_indices]
            selected_xyz = all_xyz.view([-1, 3])[candidate_mask]
            selected_grid_coords = torch.round((selected_xyz-self.init_pos)/selected_xyz_size).int()
            selected_grid_coords_unique, inverse_indices = torch.unique(selected_grid_coords, return_inverse=True, dim=0)
            selected_xyz_size_unique = torch.zeros_like(selected_grid_coords_unique, dtype=selected_xyz_size.dtype)
            group_sizes = torch.bincount(inverse_indices, minlength=selected_grid_coords_unique.shape[0])
            selected_xyz_size_sum = torch.zeros_like(selected_xyz_size_unique).scatter_add_(0, inverse_indices.unsqueeze(-1).expand(-1, selected_xyz_size.size(-1)), selected_xyz_size)
            selected_xyz_size_unique = selected_xyz_size_sum / group_sizes.unsqueeze(-1)
            selected_xyz_size_unique[group_sizes == 0] = 0
            selected_xyz_size_unique = selected_xyz_size_sum / group_sizes.unsqueeze(-1)
            if selected_grid_coords_unique.shape[0] > 0 and grid_coords.shape[0] > 0:
                remove_duplicates = self.get_remove_duplicates(grid_coords, selected_grid_coords_unique)
                remove_duplicates = ~remove_duplicates
                candidate_anchor = selected_grid_coords_unique[remove_duplicates]*selected_xyz_size_unique[remove_duplicates]+self.init_pos
                new_level = torch.ones(candidate_anchor.shape[0], dtype=torch.int, device='cuda') * cur_level
                candidate_anchor, new_level, _, weed_mask = self.weed_out(candidate_anchor, new_level)
                remove_duplicates_clone = remove_duplicates.clone()
                remove_duplicates[remove_duplicates_clone] = weed_mask
            else:
                candidate_anchor = torch.zeros([0, 3], dtype=torch.float, device='cuda')
                remove_duplicates = torch.ones([0], dtype=torch.bool, device='cuda')
                new_level = torch.zeros([0], dtype=torch.int, device='cuda')

            if (~self.progressive or iteration > self.coarse_intervals[-1]) and cur_level < self.levels - 1:
                grid_coords_ds = torch.round((self.get_anchor[level_ds_mask]-self.init_pos)/adjusted_ds_size[level_ds_mask]).int()
                selected_xyz_ds = all_xyz.view([-1, 3])[candidate_ds_mask]
                candidate_ds_indices = torch.nonzero(candidate_ds_mask, as_tuple=False).squeeze(1)
                selected_anchor_ds_indices = candidate_ds_indices // self.n_offsets
                selected_xyz_ds_size = adjusted_ds_size[selected_anchor_ds_indices]
                selected_grid_coords_ds = torch.round((selected_xyz_ds-self.init_pos)/selected_xyz_ds_size).int()
                selected_grid_coords_unique_ds, inverse_indices_ds = torch.unique(selected_grid_coords_ds, return_inverse=True, dim=0)
                selected_xyz_ds_size_unique = torch.zeros_like(selected_grid_coords_unique_ds, dtype=selected_xyz_ds_size.dtype)
                ds_group_sizes = torch.bincount(inverse_indices_ds, minlength=selected_grid_coords_unique_ds.shape[0])
                selected_xyz_ds_size_sum = torch.zeros_like(selected_xyz_ds_size_unique).scatter_add_(0, inverse_indices_ds.unsqueeze(-1).expand(-1, selected_xyz_ds_size.size(-1)), selected_xyz_ds_size)
                selected_xyz_ds_size_unique = selected_xyz_ds_size_sum / ds_group_sizes.unsqueeze(-1)
                selected_xyz_ds_size_unique[ds_group_sizes == 0] = 0
                selected_xyz_ds_size_unique = selected_xyz_ds_size_sum / ds_group_sizes.unsqueeze(-1)
                if selected_grid_coords_unique_ds.shape[0] > 0 and grid_coords_ds.shape[0] > 0:
                    remove_duplicates_ds = self.get_remove_duplicates(grid_coords_ds, selected_grid_coords_unique_ds)
                    remove_duplicates_ds = ~remove_duplicates_ds
                    candidate_anchor_ds = selected_grid_coords_unique_ds[remove_duplicates_ds]*selected_xyz_ds_size_unique[remove_duplicates_ds]+self.init_pos
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

                new_feat = self._anchor_feat[selected_anchor_indices]
                new_feat = scatter_max(new_feat, inverse_indices.unsqueeze(1).expand(-1, new_feat.size(1)), dim=0)[0][remove_duplicates]
                new_feat_ds = torch.zeros([candidate_anchor_ds.shape[0], self.feat_dim], dtype=torch.float, device='cuda')
                new_feat = torch.cat([new_feat, new_feat_ds], dim=0)
                new_labels = self.compute_region_labels(candidate_anchor)
                new_labels_ds = self.compute_region_labels(candidate_anchor_ds)
                # cur_size / ds_size ,
                cur_size_scalar = float(cur_size.detach().max().item()) if torch.is_tensor(cur_size) else float(cur_size)
                ds_size_scalar = float(ds_size.detach().max().item()) if torch.is_tensor(ds_size) else float(ds_size)
                new_adjusted_cur_size = torch.full(candidate_anchor.shape, cur_size_scalar, dtype=torch.float, device=candidate_anchor.device)
                new_adjusted_ds_size = torch.full(candidate_anchor_ds.shape, ds_size_scalar, dtype=torch.float, device=candidate_anchor_ds.device)
                new_adjusted_cur_size[new_labels == 0] = self.dynamic_voxel_sizes["near"] / (float(self.fork) ** cur_level)
                new_adjusted_cur_size[new_labels == 1] = self.dynamic_voxel_sizes["mid"] / (float(self.fork) ** cur_level)
                new_adjusted_cur_size[new_labels == 2] = self.dynamic_voxel_sizes["far"] / (float(self.fork) ** cur_level)
                new_adjusted_ds_size = torch.full(candidate_anchor_ds.shape, ds_size_scalar, dtype=torch.float, device=candidate_anchor_ds.device)
                new_adjusted_ds_size[new_labels_ds == 0] = self.dynamic_voxel_sizes["near"] / (float(self.fork) ** (cur_level + 1)) / self.fork
                new_adjusted_ds_size[new_labels_ds == 1] = self.dynamic_voxel_sizes["mid"] / (float(self.fork) ** (cur_level + 1)) / self.fork
                new_adjusted_ds_size[new_labels_ds == 2] = self.dynamic_voxel_sizes["far"] / (float(self.fork) ** (cur_level + 1)) / self.fork

                new_scaling = (torch.ones_like(candidate_anchor).float().cuda()*new_adjusted_cur_size).repeat([1,2])
                new_scaling_ds = (torch.ones_like(candidate_anchor_ds).float().cuda()*new_adjusted_ds_size).repeat([1,2])
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

    def save_mlp_checkpoints(self, path, mode = 'split', model_name=''):#split or unite
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


    def load_mlp_checkpoints(self, path, mode = 'split', model_name=''):#split or unite
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
        normal = super().get_normal
        return normal if self.background_mask is None else normal[self.background_mask]

    @property
    def get_semantic(self):
        semantic = super().get_semantic
        return semantic if self.background_mask is None else semantic[self.background_mask]

    def set_background_mask(self, camera: Camera):
        pass


    @property
    def get_scaling(self):
        scaling = super().get_scaling
        return scaling if self.background_mask is None else scaling[self.background_mask]

    @property
    def get_rotation(self):
        rotation = super().get_rotation
        return rotation if self.background_mask is None else rotation[self.background_mask]

    @property
    def get_anchor(self):
        anchor = super().get_anchor
        return anchor if self.background_mask is None else anchor[self.background_mask]

    @property
    def get_features(self):
        features = super().get_features
        return features if self.background_mask is None else features[self.background_mask]

    @property
    def get_opacity(self):
        opacity = super().get_opacity
        return opacity if self.background_mask is None else opacity[self.background_mask]

    @property
    def get_semantic(self):
        semantic = super().get_semantic
        return semantic if self.background_mask is None else semantic[self.background_mask]

    def update_optimizer(self):
        #  iteration ,(10M+ )

        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
