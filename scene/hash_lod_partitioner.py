"""
Hash-Based Learnable LOD + Adaptive Partitioning

:Hash,LOD
- Hash
- near/mid/far
- RLLOD

:
1. HashLOD
2. Hash level,
3. RLLOD bias
"""

import torch
import torch.nn as nn
import numpy as np
import math
from typing import Dict, Tuple, Optional, List
from sklearn.mixture import GaussianMixture


class HashLODPartitioner(nn.Module):
    """
    Hash-Based LOD Partitioner

    Hash:
    - Hash encoding
    - /
    - Hash level
    - RLLOD
    """

    def __init__(
        self,
        n_lod_levels: int = 8,
        n_hash_levels: int = 12,
        hash_features_per_level: int = 2,
        base_voxel_size: float = 0.4,
        fork: float = 2.0,
        n_regions: int = 3,  # near, mid, far
        region_names: List[str] = None,
        use_learnable_lod_bias: bool = True,
        device: str = 'cuda'
    ):
        super().__init__()

        self.n_lod_levels = n_lod_levels
        self.n_hash_levels = n_hash_levels
        self.hash_features_per_level = hash_features_per_level
        self.base_voxel_size = base_voxel_size
        self.fork = fork
        self.n_regions = n_regions
        self.region_names = region_names or ["near", "mid", "far"]
        self.use_learnable_lod_bias = use_learnable_lod_bias
        self.device = device


        # bounds[0]: near-mid, bounds[1]: mid-far
        self._bounds = nn.Parameter(
            torch.tensor([0.0, 0.0], dtype=torch.float32, device=device),
            requires_grad=False
        )

        # Hash level [start_level, end_level]

        #   "",RLHash
        self.region_hash_levels = nn.ParameterDict({
            "near": nn.Parameter(torch.tensor([n_hash_levels//3, n_hash_levels], dtype=torch.float32, device=device), requires_grad=False),  # Hash
            "mid": nn.Parameter(torch.tensor([n_hash_levels//4, n_hash_levels*3//4], dtype=torch.float32, device=device), requires_grad=False),
            "far": nn.Parameter(torch.tensor([0, n_hash_levels*2//3], dtype=torch.float32, device=device), requires_grad=False),  # Hash
        })

        #  (anchor)


        #   - mid:



        self.region_voxel_scales = nn.ParameterDict({
            "near": nn.Parameter(torch.tensor(0.8, dtype=torch.float32, device=device), requires_grad=False),
            "mid": nn.Parameter(torch.tensor(1.0, dtype=torch.float32, device=device), requires_grad=False),
            "far": nn.Parameter(torch.tensor(1.5, dtype=torch.float32, device=device), requires_grad=False),
        })

        # LOD levelHash level ()
        self._lod_to_hash_weights = nn.Parameter(
            self._init_lod_to_hash_mapping(),
            requires_grad=False
        )

        # LOD bias per anchor (Eq.12)
        # anchors
        self._lod_bias = None
        self._opacity_scale = None


        self.main_direction = None  # PCA
        self.near_is_low = True
        self.xyz_min = None
        self.xyz_max = None


        self.region_anchor_counts = {}
        self.total_anchors = 0

    def _init_lod_to_hash_mapping(self) -> torch.Tensor:
        """LOD levelHash level"""
        # soft mapping [n_lod_levels, n_hash_levels]
        # LOD levelhash levels
        weights = torch.zeros(self.n_lod_levels, self.n_hash_levels, device=self.device)

        for lod_lv in range(self.n_lod_levels):
            # hash level
            center_hash_lv = lod_lv * (self.n_hash_levels - 1) / max(1, self.n_lod_levels - 1)

            for hash_lv in range(self.n_hash_levels):
                dist = abs(hash_lv - center_hash_lv)
                weights[lod_lv, hash_lv] = math.exp(-dist * dist / 2.0)


            weights[lod_lv] = weights[lod_lv] / weights[lod_lv].sum()

        return weights

    @property
    def bounds(self) -> List[float]:
        """"""
        return [self._bounds[0].item(), self._bounds[1].item()]

    @bounds.setter
    def bounds(self, value: List[float]):
        """"""
        self._bounds.data = torch.tensor(value, dtype=torch.float32, device=self.device)

    def compute_depth_projection(
        self,
        points: torch.Tensor,
        init_pos: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        ()

        Args:
            points: [N, 3]
            init_pos: [3]

        Returns:
            proj_coords: [N]
            main_direction: [3]
        """
        if init_pos is None:
            init_pos = points.mean(dim=0)

        # PCA
        centered_data = points - points.mean(dim=0)
        try:
            U, S, V = torch.pca_lowrank(centered_data, q=min(3, centered_data.shape[1]))
            main_direction = V[:, 0]
        except Exception:
            # Fallback: z
            main_direction = torch.tensor([0.0, 0.0, 1.0], device=points.device)

        self.main_direction = main_direction


        proj_coords = torch.matmul(points - init_pos, main_direction)

        return proj_coords, main_direction

    def determine_near_far_direction(
        self,
        proj_coords: torch.Tensor,
        first_cam_center: torch.Tensor = None,
        points: torch.Tensor = None
    ) -> bool:
        """


        Returns:
            near_is_low: True
        """
        if first_cam_center is not None and points is not None and self.main_direction is not None:

            cam_proj = torch.dot(first_cam_center - points.mean(dim=0), self.main_direction)
            proj_median = torch.median(proj_coords)
            self.near_is_low = cam_proj.item() < proj_median.item()
        else:

            q10 = torch.quantile(proj_coords, 0.1)
            q90 = torch.quantile(proj_coords, 0.9)
            mid_point = (q10 + q90) / 2
            left_density = (proj_coords > mid_point).float().mean()
            right_density = (proj_coords <= mid_point).float().mean()
            self.near_is_low = left_density <= right_density

        return self.near_is_low

    def estimate_initial_bounds(
        self,
        proj_coords: torch.Tensor,
        method: str = 'gmm'
    ) -> List[float]:
        """


        Args:
            proj_coords: [N]
            method: 'gmm' ()  'quantile' ()
        """
        if method == 'gmm':
            return self._gmm_segmentation(proj_coords)
        else:
            return self._quantile_segmentation(proj_coords)

    def _gmm_segmentation(self, proj_coords: torch.Tensor) -> List[float]:
        """GMM"""
        proj_np = proj_coords.detach().cpu().numpy().reshape(-1, 1)

        try:
            gmm = GaussianMixture(n_components=3, random_state=42, n_init=3)
            gmm.fit(proj_np)

            means = gmm.means_.flatten()
            variances = gmm.covariances_.flatten()


            sorted_idx = np.argsort(means)
            means_sorted = means[sorted_idx]


            if self.near_is_low:
                near_idx, mid_idx, far_idx = 0, 1, 2
            else:
                near_idx, mid_idx, far_idx = 2, 1, 0

            near_mean = means_sorted[near_idx]
            mid_mean = means_sorted[mid_idx]
            far_mean = means_sorted[far_idx]


            near_bound = (near_mean + mid_mean) / 2
            far_bound = (mid_mean + far_mean) / 2

        except Exception as e:
            print(f"[HashLODPartitioner] GMM failed: {e}, using quantile fallback")
            return self._quantile_segmentation(proj_coords)

        return [float(near_bound), float(far_bound)]

    def _quantile_segmentation(self, proj_coords: torch.Tensor) -> List[float]:
        """"""
        q10 = torch.quantile(proj_coords, 0.1).item()
        q90 = torch.quantile(proj_coords, 0.9).item()

        range_val = q90 - q10
        if self.near_is_low:
            near_bound = q10 + range_val / 3
            far_bound = q90 - range_val / 3
        else:
            near_bound = q90 - range_val / 3
            far_bound = q10 + range_val / 3

        return [near_bound, far_bound]

    def partition_and_sample(
        self,
        points: torch.Tensor,
        xyz_min: torch.Tensor,
        xyz_max: torch.Tensor,
        init_pos: torch.Tensor = None,
        first_cam_center: torch.Tensor = None,
        compute_new_bounds: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        :anchors

         :
        1. ****:LOD(0n_lod_levels-1),
        2. ****:region_voxel_scales(near,far)
        3. **LOD**:LOD(set_anchor_mask)
           - far,anchorsLOD
           - farLOD
        4. **Hash level**:Hash level(),
           LOD

        Args:
            points: [N, 3]
            xyz_min, xyz_max:
            init_pos:
            first_cam_center: ()
            compute_new_bounds:

        Returns:
            dict containing:
                - positions: [M, 3] anchor
                - levels: [M] LOD()
                - region_labels: [M]  (0=near, 1=mid, 2=far)
                - bounds: [2]
        """
        self.xyz_min = xyz_min
        self.xyz_max = xyz_max

        if init_pos is None:
            init_pos = xyz_min.clone()


        proj_coords, main_direction = self.compute_depth_projection(points, init_pos)


        self.determine_near_far_direction(proj_coords, first_cam_center, points)


        if compute_new_bounds or self._bounds[0].item() == 0.0:
            bounds = self.estimate_initial_bounds(proj_coords)
            self.bounds = bounds
        else:
            bounds = self.bounds

        # 4. anchors
        all_positions = []
        all_levels = []
        all_region_labels = []

        near_is_low = self.near_is_low

        for region_idx, region in enumerate(self.region_names):
            # mask
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
            else:  # far
                if near_is_low:
                    mask = proj_coords >= bounds[1]
                else:
                    mask = proj_coords <= bounds[1]

            if not mask.any():
                self.region_anchor_counts[region] = 0
                continue

            region_points = points[mask]


            voxel_scale = self.region_voxel_scales[region].item()
            region_base_voxel = self.base_voxel_size * voxel_scale


            if region_idx == 0:
                print(f"[HashLODPartitioner] :")
                print(f"  - base_voxel_size: {self.base_voxel_size:.4f}")
                print(f"  - n_lod_levels: {self.n_lod_levels}")
                for r in self.region_names:
                    scale = self.region_voxel_scales[r].item()
                    print(f"  - {r}: voxel_scale={scale:.2f}, region_base_voxel={self.base_voxel_size * scale:.4f}")

            # Hash level(Hash encoding,LOD)
            hash_level_range = self.region_hash_levels[region].data
            min_hash_level = int(hash_level_range[0].item())
            max_hash_level = int(hash_level_range[1].item())


            #   1. :LOD(0n_lod_levels-1),
            #   2. :region_voxel_scales(near,far)
            #   3. :LOD(set_anchor_mask)
            #   4. Hash level:Hash level,LOD
            lod_start = 0
            lod_end = self.n_lod_levels

            region_positions = []
            region_levels = []
            for lod_level in range(lod_start, lod_end):
                cur_voxel_size = region_base_voxel / (self.fork ** lod_level)


                quantized = torch.round((region_points - init_pos) / cur_voxel_size) * cur_voxel_size + init_pos
                unique_pos = torch.unique(quantized, dim=0)

                region_positions.append(unique_pos)
                level_tensor = torch.full((unique_pos.shape[0],), lod_level, dtype=torch.int, device=points.device)
                region_levels.append(level_tensor)

            if region_positions:
                region_pos_cat = torch.cat(region_positions, dim=0)
                region_level_cat = torch.cat(region_levels, dim=0)
                region_label = torch.full((region_pos_cat.shape[0],), region_idx, dtype=torch.int, device=points.device)

                all_positions.append(region_pos_cat)
                all_levels.append(region_level_cat)
                all_region_labels.append(region_label)

                self.region_anchor_counts[region] = region_pos_cat.shape[0]


        if all_positions:
            positions = torch.cat(all_positions, dim=0)
            levels = torch.cat(all_levels, dim=0)
            region_labels = torch.cat(all_region_labels, dim=0)
        else:
            positions = torch.empty(0, 3, device=points.device)
            levels = torch.empty(0, dtype=torch.int, device=points.device)
            region_labels = torch.empty(0, dtype=torch.int, device=points.device)

        self.total_anchors = positions.shape[0]

        # 5. LOD bias
        if self.use_learnable_lod_bias and self.total_anchors > 0:
            self._lod_bias = nn.Parameter(
                torch.zeros(self.total_anchors, device=points.device),
                requires_grad=True
            )
            self._opacity_scale = torch.ones(self.total_anchors, device=points.device)

        print(f"[HashLODPartitioner] Partition complete:")
        print(f"  - Total anchors: {self.total_anchors}")
        for region, count in self.region_anchor_counts.items():
            print(f"  - {region}: {count} anchors")
        print(f"  - Bounds: near={bounds[0]:.2f}, far={bounds[1]:.2f}")
        print(f"  - Near is low: {near_is_low}")


        del all_positions, all_levels, all_region_labels
        import gc
        gc.collect()
        torch.cuda.empty_cache()

        return {
            'positions': positions,
            'levels': levels,
            'region_labels': region_labels,
            'bounds': bounds,
            'main_direction': main_direction,
            'near_is_low': near_is_low
        }

    def get_hash_level_for_lod(self, lod_level: int, region: str = None) -> Tuple[int, int]:
        """
        LOD levelHash level

        Args:
            lod_level: LOD
            region: ,

        Returns:
            (min_hash_level, max_hash_level)
        """
        if region is not None and region in self.region_hash_levels:

            range_tensor = self.region_hash_levels[region].data
            return int(range_tensor[0].item()), int(range_tensor[1].item())


        weights = self._lod_to_hash_weights[lod_level]
        # hash levels
        significant = weights > 0.1
        if significant.any():
            indices = torch.where(significant)[0]
            return int(indices.min().item()), int(indices.max().item())

        # Fallback
        center = int(lod_level * (self.n_hash_levels - 1) / max(1, self.n_lod_levels - 1))
        return max(0, center - 1), min(self.n_hash_levels - 1, center + 1)

    def get_lod_bias_adjusted_level(
        self,
        anchor_indices: torch.Tensor,
        base_levels: torch.Tensor,
        distances: torch.Tensor,
        d_max: float
    ) -> torch.Tensor:
        """
        LOD bias(Eq.12)

        U_b = {i | L_i <= log_s(d_max/d_i) + _i}

        Args:
            anchor_indices: anchor
            base_levels: LOD
            distances:
            d_max:

        Returns:
            adjusted_levels:
        """
        if self._lod_bias is None:
            return base_levels.float()
        log_ratio = torch.log(d_max / (distances + 1e-6)) / math.log(self.fork)

        # LOD bias
        biases = self._lod_bias[anchor_indices]
        adjusted = log_ratio + biases

        return adjusted

    def get_opacity_scale(self, anchor_indices: torch.Tensor) -> torch.Tensor:
        """opacity(Eq.13)"""
        if self._opacity_scale is None:
            return torch.ones(anchor_indices.shape[0], device=anchor_indices.device)
        return self._opacity_scale[anchor_indices]

    def get_state_features(self) -> torch.Tensor:
        """Return compact partition state features."""
        features = []


        bounds = self.bounds
        if self.xyz_min is not None and self.xyz_max is not None:
            scene_range = (self.xyz_max - self.xyz_min).max().item()
            features.append(bounds[0] / scene_range if scene_range > 0 else 0.0)
            features.append(bounds[1] / scene_range if scene_range > 0 else 0.0)
        else:
            features.extend([0.0, 0.0])

        # anchor
        total = max(1, self.total_anchors)
        for region in self.region_names:
            count = self.region_anchor_counts.get(region, 0)
            features.append(count / total)

        # hash level
        for region in self.region_names:
            range_tensor = self.region_hash_levels[region].data
            features.append(range_tensor[0].item() / self.n_hash_levels)
            features.append(range_tensor[1].item() / self.n_hash_levels)


        for region in self.region_names:
            features.append(self.region_voxel_scales[region].item())

        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def save_state(self) -> Dict:
        """"""
        return {
            'bounds': self.bounds,
            'region_hash_levels': {k: v.data.cpu().numpy().tolist() for k, v in self.region_hash_levels.items()},
            'region_voxel_scales': {k: v.item() for k, v in self.region_voxel_scales.items()},
            'lod_bias': self._lod_bias.data.cpu().numpy() if self._lod_bias is not None else None,
            'opacity_scale': self._opacity_scale.cpu().numpy() if self._opacity_scale is not None else None,
            'main_direction': self.main_direction.cpu().numpy() if self.main_direction is not None else None,
            'near_is_low': self.near_is_low,
        }

    def load_state(self, state: Dict):
        """"""
        if 'bounds' in state:
            self.bounds = state['bounds']
        if 'region_hash_levels' in state:
            for k, v in state['region_hash_levels'].items():
                if k in self.region_hash_levels:
                    self.region_hash_levels[k].data = torch.tensor(v, dtype=torch.float32, device=self.device)
        if 'region_voxel_scales' in state:
            for k, v in state['region_voxel_scales'].items():
                if k in self.region_voxel_scales:
                    self.region_voxel_scales[k].data = torch.tensor(v, dtype=torch.float32, device=self.device)
        if 'lod_bias' in state and state['lod_bias'] is not None:
            self._lod_bias = nn.Parameter(
                torch.tensor(state['lod_bias'], dtype=torch.float32, device=self.device),
                requires_grad=True
            )
        if 'opacity_scale' in state and state['opacity_scale'] is not None:
            self._opacity_scale = torch.tensor(state['opacity_scale'], dtype=torch.float32, device=self.device)
        if 'main_direction' in state and state['main_direction'] is not None:
            self.main_direction = torch.tensor(state['main_direction'], dtype=torch.float32, device=self.device)
        if 'near_is_low' in state:
            self.near_is_low = state['near_is_low']


def create_hash_lod_partitioner(
    args,
    device: str = 'cuda'
) -> HashLODPartitioner:
    """
    HashLODPartitioner
    """
    n_lod_levels = getattr(args, 'levels', 8)
    n_hash_levels = getattr(args, 'hash_levels', 12)
    hash_features = getattr(args, 'hash_features_per_level', 2)
    base_voxel = getattr(args, 'base_voxel_size', 0.4)
    fork = getattr(args, 'fork', 2.0)
    use_lod_bias = getattr(args, 'use_lod_bias', True)

    return HashLODPartitioner(
        n_lod_levels=n_lod_levels,
        n_hash_levels=n_hash_levels,
        hash_features_per_level=hash_features,
        base_voxel_size=base_voxel,
        fork=fork,
        use_learnable_lod_bias=use_lod_bias,
        device=device
    )
