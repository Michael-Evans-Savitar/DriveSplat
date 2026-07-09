import torch
from scene.anchor_deformation import AnchorDeformationNetwork, AnchorToGaussianTransform
import os
from utils.system_utils import searchForMaxIteration
from utils.general_utils import get_expon_lr_func


class DeformModel:
    def __init__(self, is_blender=False, is_6dof=False, max_lod_level=32,
                 use_anchor_deform=True,
                 anchor_deform_feat_dim=32, anchor_deform_hidden=256,
                 anchor_deform_layers=8, anchor_deform_use_rotation=True,
                 anchor_deform_xyz_multires=10, anchor_deform_t_multires=10):
        self.max_lod_level = max_lod_level

        if not use_anchor_deform:
            raise ValueError("DriveSplat release expects anchor-conditioned deformation to be enabled")

        print(f"\n{'='*70}")
        print("[DeformModel] Using Anchor-Conditioned Deformation")
        print("  - Deformation is predicted in anchor space and propagated to Gaussians")
        print(f"{'='*70}\n")
        self.deform = AnchorDeformationNetwork(
            xyz_multires=anchor_deform_xyz_multires,
            t_multires=anchor_deform_t_multires,
            feat_dim=anchor_deform_feat_dim,
            hidden_dim=anchor_deform_hidden,
            num_layers=anchor_deform_layers,
            output_rotation=anchor_deform_use_rotation,
        ).cuda()
        self.anchor_transform = AnchorToGaussianTransform(
            use_rotation=anchor_deform_use_rotation
        ).cuda()

        self.optimizer = None
        self.spatial_lr_scale = 5
        self.last_reg = None

    def step(self, xyz, time_emb, level=None, use_euler=True, dt=None,
             anchor=None, offsets=None, scaling=None, rotation=None, non_rigid_weight=1.0,
             anchor_feat=None, anchor_to_gaussian_idx=None):
        """Compute deformation residuals for the active deformation module."""
        if time_emb.numel() == 0:
            device = xyz.device if xyz is not None else time_emb.device
            dtype = xyz.dtype if xyz is not None else time_emb.dtype
            n = 0 if xyz is None else xyz.shape[0]
            return (
                torch.zeros((n, 3), device=device, dtype=dtype),
                torch.zeros((n, 4), device=device, dtype=dtype),
                torch.zeros((n, 3), device=device, dtype=dtype),
                None,
            )

        if isinstance(self.deform, AnchorDeformationNetwork):
            if anchor is None or anchor_feat is None or offsets is None:
                raise ValueError("anchor, anchor_feat and offsets are required for AnchorDeformationNetwork")
            if anchor_to_gaussian_idx is None:
                raise ValueError("anchor_to_gaussian_idx is required for AnchorDeformationNetwork")

            if time_emb.dim() == 1:
                time_for_anchor = time_emb.view(-1, 1)
            else:
                time_for_anchor = time_emb
            if time_for_anchor.shape[0] != anchor.shape[0]:
                time_for_anchor = time_for_anchor[:1].expand(anchor.shape[0], -1)

            anchor_output = self.deform(anchor, anchor_feat, time_for_anchor)
            d_anchor = anchor_output["d_anchor"]
            R_anchor = anchor_output.get("R_anchor", None)

            gaussian_xyz = self.anchor_transform(
                anchor, offsets, d_anchor, R_anchor, anchor_to_gaussian_idx
            )
            d_xyz = gaussian_xyz - (anchor[anchor_to_gaussian_idx] + offsets)
            d_xyz = d_xyz * non_rigid_weight

            d_rotation = torch.zeros(xyz.shape[0], 4, device=xyz.device, dtype=xyz.dtype)
            d_scaling = torch.zeros(xyz.shape[0], 3, device=xyz.device, dtype=xyz.dtype)
            return d_xyz, d_rotation, d_scaling, None

        raise TypeError(f"Unsupported deformation module: {type(self.deform).__name__}")

    def train_setting(self, training_args):
        deform_lr_init = training_args.deformation_lr_init * self.spatial_lr_scale
        deform_lr_final = training_args.deformation_lr_final * self.spatial_lr_scale
        params = [{
            'params': list(self.deform.parameters()),
            'lr': deform_lr_init,
            'name': 'deform',
        }]
        self.optimizer = torch.optim.Adam(params, lr=0.0, eps=1e-15)

        self.deform_scheduler_args = get_expon_lr_func(
            lr_init=deform_lr_init,
            lr_final=deform_lr_final,
            lr_delay_mult=training_args.deformation_lr_delay_mult,
            max_steps=training_args.deform_lr_max_steps,
        )

    def save_weights(self, model_path, iteration, model_name):
        out_weights_path = os.path.join(model_path, "deform/iteration_{}".format(iteration))
        os.makedirs(out_weights_path, exist_ok=True)
        torch.save(self.deform.state_dict(), os.path.join(out_weights_path, "{}_deform.pth".format(model_name)))

    def load_weights(self, model_path, iteration=-1, model_name="obj_001"):
        if iteration == -1:
            loaded_iter = searchForMaxIteration(os.path.join(model_path, "deform"))
        else:
            loaded_iter = iteration

        base_path = os.path.join(model_path, "deform/iteration_{}".format(loaded_iter))
        deform_path = os.path.join(base_path, "{}_deform.pth".format(model_name))
        if os.path.exists(deform_path):
            self.deform.load_state_dict(torch.load(deform_path))
        else:
            print(f"Warning: Deform weights not found at {deform_path}")

    def update_learning_rate(self, iteration):
        last_lr = None
        for param_group in self.optimizer.param_groups:
            if param_group["name"] == "deform":
                lr = self.deform_scheduler_args(iteration)
                param_group["lr"] = lr
                last_lr = lr
        return last_lr
