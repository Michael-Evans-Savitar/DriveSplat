"""Anchor-conditioned deformation for neural Gaussian actors."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.time_utils import get_embedder


class AnchorDeformationNetwork(nn.Module):
    """Predict deformation at the anchor level and propagate it to Gaussians."""

    def __init__(self,
                 xyz_multires=10,
                 t_multires=10,
                 feat_dim=32,
                 hidden_dim=256,
                 num_layers=8,
        output_rotation=True,
        use_skip=True):
        super().__init__()

        self.output_rotation = output_rotation
        self.use_skip = use_skip

        self.embed_xyz, xyz_dim = get_embedder(xyz_multires, 3)
        self.embed_time, time_dim = get_embedder(t_multires, 1)

        input_dim = xyz_dim + feat_dim + time_dim

        print(f"\n{'='*70}")
        print(f"[AnchorDeformationNetwork]  Anchor-Conditioned Deformation")
        print(f"  - XYZ encoding: {xyz_dim} dims (multires={xyz_multires})")
        print(f"  - Anchor feature: {feat_dim} dims (scene semantics!)")
        print(f"  - Time encoding: {time_dim} dims (t_multires={t_multires})")
        print(f"  - Total input: {input_dim} dims")
        print(f"  - Network: {num_layers} layers x {hidden_dim} hidden")
        print(f"  - Output rotation: {output_rotation}")
        print("  - Computes deformation on anchors and shares it with attached Gaussians")
        print(f"{'='*70}\n")

        self.skips = [num_layers // 2] if use_skip else []
        layers = []

        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            elif i in self.skips:
                layers.append(nn.Linear(hidden_dim + input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))

        self.mlp = nn.Sequential(*layers)

        self.displacement_head = nn.Linear(hidden_dim, 3)

        if output_rotation:
            self.rotation_head = nn.Linear(hidden_dim, 6)

        self.disp_scale = nn.Parameter(torch.tensor(1.0))

        nn.init.xavier_uniform_(self.displacement_head.weight, gain=0.01)
        nn.init.zeros_(self.displacement_head.bias)
        if output_rotation:
            nn.init.xavier_uniform_(self.rotation_head.weight, gain=0.01)
            nn.init.constant_(self.rotation_head.bias, 0.0)

    def forward(self, anchor_xyz, anchor_feat, time):
        """
        Args:
            anchor_xyz: (Na, 3) - anchor positions in canonical space
            anchor_feat: (Na, F) - anchor features (learned scene semantics)
            time: (Na, 1) or (1,) - normalized time [0, 1]

        Returns:
            dict with keys:
                - d_anchor: (Na, 3) - anchor displacement
                - R_anchor: (Na, 3, 3) - local rotation for offset transformation
                - feat_norm: scalar - feature magnitude (for analysis)
        """
        Na = anchor_xyz.shape[0]

        if time.dim() == 0:
            time = time.view(1, 1).expand(Na, 1)
        elif time.dim() == 1:
            time = time.view(-1, 1)
        if time.shape[0] == 1 and Na > 1:
            time = time.expand(Na, 1)

        xyz_emb = self.embed_xyz(anchor_xyz)
        time_emb = self.embed_time(time)

        h = torch.cat([xyz_emb, anchor_feat, time_emb], dim=-1)
        skip_input = h

        for i, layer in enumerate(self.mlp):
            if isinstance(layer, nn.Linear):
                if i // 2 in self.skips and i > 0:
                    h = torch.cat([skip_input, h], dim=-1)
            h = layer(h)

        d_anchor = self.displacement_head(h) * self.disp_scale

        if self.output_rotation:
            # 6D rotation representation (more stable than quaternion or euler angles)
            # Reference: "On the Continuity of Rotation Representations in Neural Networks"
            rot_6d = self.rotation_head(h)
            R_anchor = rotation_6d_to_matrix(rot_6d)
        else:
            R_anchor = None

        feat_norm = anchor_feat.norm(dim=-1).mean()

        return {
            "d_anchor": d_anchor,
            "R_anchor": R_anchor,
            "feat_norm": feat_norm
        }


def rotation_6d_to_matrix(rot_6d):
    """Convert the continuous 6D rotation representation to rotation matrices."""
    a1 = rot_6d[:, :3]
    a2 = rot_6d[:, 3:]

    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(dim=-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)

    return torch.stack([b1, b2, b3], dim=-1)


class AnchorToGaussianTransform(nn.Module):
    """Apply anchor deformation to Gaussian offsets."""

    def __init__(self, use_rotation=True, rotation_weight=1.0):
        super().__init__()
        self.use_rotation = use_rotation
        self.rotation_weight = rotation_weight

    def forward(self, anchor_xyz_canon, offset_canon, d_anchor, R_anchor=None,
                anchor_to_gaussian_idx=None):
        """
        Args:
            anchor_xyz_canon: (Na, 3) - anchor positions in canonical space
            offset_canon: (Ng, 3) - offsets in canonical space
            d_anchor: (Na, 3) - anchor displacement
            R_anchor: (Na, 3, 3) - anchor local rotation (optional)
            anchor_to_gaussian_idx: (Ng,) - mapping from gaussian to its anchor

        Returns:
            gaussian_xyz_deformed: (Ng, 3) - deformed gaussian positions
        """
        Ng = offset_canon.shape[0]

        if anchor_to_gaussian_idx is not None:
            anchor_xyz_per_gaussian = anchor_xyz_canon[anchor_to_gaussian_idx]
            d_anchor_per_gaussian = d_anchor[anchor_to_gaussian_idx]
        else:
            anchor_xyz_per_gaussian = anchor_xyz_canon
            d_anchor_per_gaussian = d_anchor

        anchor_xyz_deformed = anchor_xyz_per_gaussian + d_anchor_per_gaussian

        if self.use_rotation and R_anchor is not None:
            if anchor_to_gaussian_idx is not None:
                R_per_gaussian = R_anchor[anchor_to_gaussian_idx]
            else:
                R_per_gaussian = R_anchor

            offset_rotated = torch.bmm(R_per_gaussian, offset_canon.unsqueeze(-1)).squeeze(-1)

            offset_transformed = (self.rotation_weight * offset_rotated +
                                (1 - self.rotation_weight) * offset_canon)
        else:
            offset_transformed = offset_canon

        return anchor_xyz_deformed + offset_transformed
