import math
from typing import Optional, Tuple

import torch
import torch.nn as nn

# Try to import tiny-cuda-nn for optimized hash encoding
try:
    import tinycudann as tcnn
    TCNN_AVAILABLE = True
except ImportError:
    TCNN_AVAILABLE = False
    print("[HashEncoding] tiny-cuda-nn not available, using PyTorch fallback")


def _spatial_hash(coords: torch.Tensor, hash_size: int) -> torch.Tensor:
    """
    Simple spatial hash for integer grid coordinates.

    Args:
        coords: (N, 3) int tensor.
        hash_size: size of the hash table.
    Returns:
        (N,) long tensor of hashed indices in [0, hash_size).
    """
    # Large primes to mix bits. Staying in int64 to avoid overflow on cuda.
    x, y, z = coords[:, 0].long(), coords[:, 1].long(), coords[:, 2].long()
    hashed = (x * 73856093) ^ (y * 19349663) ^ (z * 83492791)
    return hashed % hash_size


class HashEncoding(nn.Module):
    """
    Instant-NGP style multiresolution hash grid encoding.

    Provides:
      - Level -> resolution mapping
      - Hash index computation
      - Optional feature interpolation per level
    """

    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 512,
        aabb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        super().__init__()
        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hash_size = 1 << log2_hashmap_size

        # Per-level resolution scaling per Instant-NGP
        self.per_level_scale = math.exp(
            math.log(finest_resolution / base_resolution) / max(n_levels - 1, 1)
        )
        self.base_resolution = base_resolution

        # AABB for normalization; default to unit cube if not provided.
        if aabb is None:
            xyz_min = torch.zeros(3)
            xyz_max = torch.ones(3)
        else:
            xyz_min, xyz_max = aabb
        self.register_buffer("xyz_min", xyz_min.float())
        self.register_buffer("xyz_max", xyz_max.float())

        # Embedding tables per level: (hash_size, n_features_per_level)
        tables = []
        for _ in range(n_levels):
            table = nn.Embedding(self.hash_size, n_features_per_level)
            nn.init.uniform_(table.weight, a=-1e-4, b=1e-4)
            tables.append(table)
        self.tables = nn.ModuleList(tables)

    def level_resolution(self, level: int) -> int:
        """Return integer resolution of a given level."""
        return int(self.base_resolution * (self.per_level_scale**level))

    def quantize_coords(self, xyz: torch.Tensor, level: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Normalize xyz to [0, 1] in AABB and quantize to integer grid coords.

        Args:
            xyz: (N, 3) float tensor in world space.
            level: level index.
        Returns:
            grid_coords: (N, 3) int tensor of grid coordinates.
            resolution: scalar int resolution for this level.
        """
        resolution = self.level_resolution(level)
        xyz_norm = (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min + 1e-8)
        # Clamp to avoid boundary issues.
        xyz_norm = torch.clamp(xyz_norm, 0.0, 0.9999)
        grid_coords = torch.floor(xyz_norm * resolution).int()
        return grid_coords, resolution

    def hash_coords(self, grid_coords: torch.Tensor) -> torch.Tensor:
        """Hash integer grid coords to table indices."""
        return _spatial_hash(grid_coords, self.hash_size)

    def lookup_features(self, xyz: torch.Tensor, use_interpolation: bool = True) -> torch.Tensor:
        """
        Query multi-resolution hash features for positions xyz.

        Args:
            xyz: (N, 3) float tensor in world coordinates.
            use_interpolation: If True, use trilinear interpolation; otherwise nearest neighbor.
        Returns:
            features: (N, n_levels * n_features_per_level)
        """
        feats = []
        for level in range(self.n_levels):
            if use_interpolation:
                feat = self._lookup_level_trilinear(xyz, level)
            else:
                grid_coords, _ = self.quantize_coords(xyz, level)
                idx = self.hash_coords(grid_coords)
                feat = self.tables[level](idx)
            feats.append(feat)
        return torch.cat(feats, dim=-1)

    def _lookup_level_trilinear(self, xyz: torch.Tensor, level: int) -> torch.Tensor:
        """
        Trilinear interpolation for a single level (Instant-NGP style).

        Args:
            xyz: (N, 3) float tensor in world space
            level: level index
        Returns:
            interpolated_feat: (N, n_features_per_level)
        """
        resolution = self.level_resolution(level)
        xyz_norm = (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min + 1e-8)
        xyz_norm = torch.clamp(xyz_norm, 0.0, 0.9999)

        # Continuous grid coordinates
        xyz_grid = xyz_norm * resolution

        # Floor and ceil for 8 corners
        xyz_floor = torch.floor(xyz_grid).int()
        xyz_ceil = xyz_floor + 1

        # Clamp ceil to avoid out-of-bounds
        xyz_ceil = torch.clamp(xyz_ceil, max=resolution - 1)

        # Interpolation weights (fractional part)
        weights = xyz_grid - xyz_floor.float()  # (N, 3) in [0, 1]

        # Query 8 corners and interpolate
        feat_interp = torch.zeros(xyz.shape[0], self.n_features_per_level,
                                   device=xyz.device, dtype=xyz.dtype)

        for i in range(2):
            for j in range(2):
                for k in range(2):
                    # Select floor or ceil for each dimension
                    corner = torch.stack([
                        xyz_floor[:, 0] if i == 0 else xyz_ceil[:, 0],
                        xyz_floor[:, 1] if j == 0 else xyz_ceil[:, 1],
                        xyz_floor[:, 2] if k == 0 else xyz_ceil[:, 2]
                    ], dim=1)

                    # Hash and lookup
                    idx = self.hash_coords(corner)
                    feat = self.tables[level](idx)

                    # Compute weight for this corner
                    w = (weights[:, 0] if i == 1 else (1 - weights[:, 0])) * \
                        (weights[:, 1] if j == 1 else (1 - weights[:, 1])) * \
                        (weights[:, 2] if k == 1 else (1 - weights[:, 2]))

                    feat_interp += feat * w.unsqueeze(-1)

        return feat_interp

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """Alias to lookup_features for nn.Module interface."""
        return self.lookup_features(xyz)

    def update_aabb(self, xyz_min: torch.Tensor, xyz_max: torch.Tensor):
        """Update bounding box (useful after loading checkpoints)."""
        self.xyz_min = xyz_min.float().to(self.xyz_min.device)
        self.xyz_max = xyz_max.float().to(self.xyz_max.device)


class TcnnHashEncoding(nn.Module):
    """
    Hash grid encoding using tiny-cuda-nn for maximum efficiency.

    This is a drop-in replacement for HashEncoding that uses NVIDIA's
    highly optimized CUDA implementation.
    """

    def __init__(
        self,
        n_levels: int = 16,
        n_features_per_level: int = 2,
        log2_hashmap_size: int = 19,
        base_resolution: int = 16,
        finest_resolution: int = 512,
        aabb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ):
        super().__init__()

        if not TCNN_AVAILABLE:
            raise RuntimeError("tiny-cuda-nn is not installed. Please install it first.")

        self.n_levels = n_levels
        self.n_features_per_level = n_features_per_level
        self.log2_hashmap_size = log2_hashmap_size
        self.hash_size = 1 << log2_hashmap_size
        self.base_resolution = base_resolution
        self.finest_resolution = finest_resolution

        # Per-level resolution scaling per Instant-NGP
        self.per_level_scale = math.exp(
            math.log(finest_resolution / base_resolution) / max(n_levels - 1, 1)
        )

        # AABB for normalization
        if aabb is None:
            xyz_min = torch.zeros(3)
            xyz_max = torch.ones(3)
        else:
            xyz_min, xyz_max = aabb
        self.register_buffer("xyz_min", xyz_min.float())
        self.register_buffer("xyz_max", xyz_max.float())

        # Create tiny-cuda-nn hash grid encoding
        # Note: tcnn expects input in [0, 1] range
        self.encoding = tcnn.Encoding(
            n_input_dims=3,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": n_levels,
                "n_features_per_level": n_features_per_level,
                "log2_hashmap_size": log2_hashmap_size,
                "base_resolution": base_resolution,
                "per_level_scale": self.per_level_scale,
            },
            dtype=torch.float32,  # Use float32 for compatibility
        )

        # Output dimension
        self.output_dim = n_levels * n_features_per_level

    def level_resolution(self, level: int) -> int:
        """Return integer resolution of a given level."""
        return int(self.base_resolution * (self.per_level_scale**level))

    def lookup_features(self, xyz: torch.Tensor, use_interpolation: bool = True) -> torch.Tensor:
        """
        Query multi-resolution hash features for positions xyz.

        Note: tiny-cuda-nn always uses trilinear interpolation (use_interpolation is ignored).

        Args:
            xyz: (N, 3) float tensor in world coordinates.
            use_interpolation: Ignored (always uses interpolation).
        Returns:
            features: (N, n_levels * n_features_per_level)
        """
        # Normalize to [0, 1] range
        xyz_norm = (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min + 1e-8)
        xyz_norm = torch.clamp(xyz_norm, 0.0, 1.0 - 1e-6)

        # Ensure contiguous and correct dtype
        xyz_norm = xyz_norm.contiguous().float()

        # Query tiny-cuda-nn encoding
        features = self.encoding(xyz_norm)

        return features.float()  # Ensure output is float32

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """Alias to lookup_features for nn.Module interface."""
        return self.lookup_features(xyz)

    def update_aabb(self, xyz_min: torch.Tensor, xyz_max: torch.Tensor):
        """Update bounding box (useful after loading checkpoints)."""
        self.xyz_min = xyz_min.float().to(self.xyz_min.device)
        self.xyz_max = xyz_max.float().to(self.xyz_max.device)

    def _lookup_level_trilinear(self, xyz: torch.Tensor, level: int) -> torch.Tensor:
        """
        Query features for a single level (for compatibility with single-level query).

        Note: tiny-cuda-nn always queries all levels, so we query all and slice.

        Args:
            xyz: (N, 3) float tensor in world space
            level: level index to query
        Returns:
            features: (N, n_features_per_level)
        """
        # Query all levels
        all_features = self.lookup_features(xyz)  # (N, n_levels * n_features_per_level)

        # Slice the features for the requested level
        start_idx = level * self.n_features_per_level
        end_idx = start_idx + self.n_features_per_level

        # Clamp level to valid range
        if level >= self.n_levels:
            level = self.n_levels - 1
            start_idx = level * self.n_features_per_level
            end_idx = start_idx + self.n_features_per_level

        return all_features[:, start_idx:end_idx]

    def quantize_coords(self, xyz: torch.Tensor, level: int):
        """
        For compatibility with PyTorch implementation.
        Not actually used by tcnn but needed for fallback code paths.
        """
        resolution = self.level_resolution(level)
        xyz_norm = (xyz - self.xyz_min) / (self.xyz_max - self.xyz_min + 1e-8)
        xyz_norm = torch.clamp(xyz_norm, 0.0, 0.9999)
        grid_coords = torch.floor(xyz_norm * resolution).int()
        return grid_coords, resolution

    def hash_coords(self, grid_coords: torch.Tensor) -> torch.Tensor:
        """For compatibility - hash integer grid coords to table indices."""
        return _spatial_hash(grid_coords, self.hash_size)

    @property
    def tables(self):
        """
        For compatibility with code that accesses .tables for parameter groups.
        Returns None as tcnn manages parameters internally.
        """
        return None


def create_hash_encoding(
    n_levels: int = 16,
    n_features_per_level: int = 2,
    log2_hashmap_size: int = 19,
    base_resolution: int = 16,
    finest_resolution: int = 512,
    aabb: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    use_tcnn: bool = True,
) -> nn.Module:
    """
    Factory function to create the appropriate hash encoding module.

    Args:
        n_levels: Number of levels in the hash grid.
        n_features_per_level: Features per level.
        log2_hashmap_size: Log2 of hash table size.
        base_resolution: Resolution at level 0.
        finest_resolution: Resolution at the finest level.
        aabb: Bounding box (xyz_min, xyz_max).
        use_tcnn: If True and available, use tiny-cuda-nn implementation.

    Returns:
        HashEncoding or TcnnHashEncoding module.
    """
    if use_tcnn and TCNN_AVAILABLE:
        print(f"[HashEncoding] Using tiny-cuda-nn (n_levels={n_levels}, features_per_level={n_features_per_level})")
        return TcnnHashEncoding(
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
            aabb=aabb,
        )
    else:
        print(f"[HashEncoding] Using PyTorch fallback (n_levels={n_levels}, features_per_level={n_features_per_level})")
        return HashEncoding(
            n_levels=n_levels,
            n_features_per_level=n_features_per_level,
            log2_hashmap_size=log2_hashmap_size,
            base_resolution=base_resolution,
            finest_resolution=finest_resolution,
            aabb=aabb,
        )
