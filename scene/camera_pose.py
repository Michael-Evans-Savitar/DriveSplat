import torch
import torch.nn as nn
from utils.camera_utils import Camera
from utils.general_utils import get_expon_lr_func, quaternion_raw_multiply, quaternion_to_matrix

class PoseCorrection(nn.Module):
    def __init__(self, metadata, args):
        super().__init__()
        self.identity_matrix = torch.eye(4).float().cuda()[:3]
        self.mode = args.pose_mode

        # per image embedding
        if self.mode == 'image':
            num_poses = metadata['num_images']
        # per frame embedding
        elif self.mode == 'frame':
            num_poses = metadata['num_frames']
        else:
            raise ValueError(f'Invalid mode: {self.mode}')

        self.pose_correction_trans = torch.nn.Parameter(torch.zeros(num_poses, 3).float().cuda()).requires_grad_(True)
        self.pose_correction_rots = torch.nn.Parameter(torch.tensor([[1, 0, 0, 0]]).repeat(num_poses, 1).float().cuda()).requires_grad_(True)

    def save_state_dict(self, is_final):
        state_dict = dict()
        state_dict['params'] = self.state_dict()
        if not is_final:
            state_dict['optimizer'] = self.optimizer.state_dict()

        return state_dict

    def load_state_dict(self, state_dict, is_training):
        super().load_state_dict(state_dict['params'])
        if is_training and 'optimizer' in state_dict:
            self.optimizer.load_state_dict(state_dict['optimizer'])

    def training_setup(self, training_args):
        pose_correction_lr_init = training_args.pose_correction_lr_init
        pose_correction_lr_final = training_args.pose_correction_lr_final
        pose_correction_max_steps = training_args.pose_correction_max_steps

        params = [
            {'params': [self.pose_correction_trans], 'lr': pose_correction_lr_init, 'name': 'pose_correction_trans'},
            {'params': [self.pose_correction_rots], 'lr': pose_correction_lr_init, 'name': 'pose_correction_rots'},
        ]

        self.optimizer = torch.optim.Adam(params=params, lr=0, eps=1e-8, weight_decay=0.01)

        self.pose_correction_scheduler_args = get_expon_lr_func(
            lr_init=pose_correction_lr_init,
            lr_final=pose_correction_lr_final,
            lr_delay_mult=training_args.pose_correction_lr_delay_mult,
            max_steps=pose_correction_max_steps,
        )

    def update_learning_rate(self, iteration):
        for param_group in self.optimizer.param_groups:
            lr = self.pose_correction_scheduler_args(iteration)
            param_group['lr'] = lr

    def update_optimizer(self):
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=None)

    def get_id(self, camera: Camera):
        if self.mode == 'image':
            return camera.uid
        elif self.mode == 'frame':
            return camera.meta['frame_idx']
        else:
            raise ValueError(f'invalid mode: {self.mode}')

    def forward(self, camera: Camera):
        id = self.get_id(camera)
        pose_correction_trans = self.pose_correction_trans[id]
        pose_correction_rot = self.pose_correction_rots[id]
        pose_correction_rot = torch.nn.functional.normalize(pose_correction_rot.unsqueeze(0))
        pose_correction_rot = quaternion_to_matrix(pose_correction_rot).squeeze(0)
        pose_correction_matrix = torch.cat([pose_correction_rot, pose_correction_trans[:, None]], dim=-1)
        padding = torch.tensor([[0, 0, 0, 1]]).float().cuda()
        pose_correction_matrix = torch.cat([pose_correction_matrix, padding], dim=0)

        return pose_correction_matrix

    def correct_gaussian_xyz(self, camera: Camera, xyz: torch.Tensor, is_training):
        # xyz: [N, 3]
        if is_training:
            id = self.get_id(camera)
            pose_correction_trans = self.pose_correction_trans[id]
            pose_correction_rot = self.pose_correction_rots[id]
            pose_correction_rot = torch.nn.functional.normalize(pose_correction_rot.unsqueeze(0), dim=-1)
            pose_correction_rot = quaternion_to_matrix(pose_correction_rot).squeeze(0)
            pose_correction_matrix = torch.cat([pose_correction_rot, pose_correction_trans[:, None]], dim=-1)
            padding = torch.tensor([[0, 0, 0, 1]]).float().cuda()
            pose_correction_matrix = torch.cat([pose_correction_matrix, padding], dim=0)
            xyz = torch.cat([xyz, torch.ones_like(xyz[..., :1])], dim=-1)
            xyz = xyz @ pose_correction_matrix.T
            xyz = xyz[:, :3]

        return xyz

    def correct_gaussian_rotation(self, camera: Camera, rotation: torch.Tensor, is_training):
        # rotation: [N, 4]
        if is_training:
            id = self.get_id(camera)
            pose_correction_rot = self.pose_correction_rots[id]
            pose_correction_rot = torch.nn.functional.normalize(pose_correction_rot.unsqueeze(0), dim=-1)
            rotation = quaternion_raw_multiply(pose_correction_rot, rotation)

        return rotation

    def regularization_loss(self):
        loss_trans = torch.abs(self.pose_correction_trans).mean()
        rots_norm = torch.nn.functional.normalize(self.pose_correction_rots, dim=-1)
        loss_rots = torch.abs(rots_norm - torch.tensor([[1, 0, 0, 0]]).float().cuda()).mean()
        loss = loss_trans + loss_rots
        return loss
