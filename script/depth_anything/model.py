import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

class DepthAnything(nn.Module):
    def __init__(self, encoder='vit_large_patch14_clip_224.openai'):
        super().__init__()
        self.encoder = timm.create_model(encoder, pretrained=False)

        hooks = {
            'vit_large_patch14_clip_224.openai': [3, 6, 9, 12]
        }

        self.hooks = hooks[encoder]
        self.act_postprocess = []
        self.feats = []

        num_blocks = len(self.encoder.blocks)
        self.hooks = [idx for idx in self.hooks if idx < num_blocks]

        for idx in self.hooks:
            self.encoder.blocks[idx].register_forward_hook(self.hook_fn_forward)

        self.decode_head = nn.ModuleList([
            nn.Conv2d(1024, 256, 3, padding=1),
            nn.Conv2d(1024, 256, 3, padding=1),
            nn.Conv2d(1024, 256, 3, padding=1),
            nn.Conv2d(1024, 256, 3, padding=1),
        ])

        self.final_conv = nn.Conv2d(1024, 1, 3, padding=1)

    def hook_fn_forward(self, module, input, output):
        self.feats.append(output)

    def forward(self, x):
        self.feats = []
        H, W = x.shape[2:]

        out = self.encoder.forward_features(x)

        features = [self.feats[0], self.feats[1], self.feats[2], self.feats[3]]
        features = [self._reshape_as_image(x) for x in features]

        features = [F.interpolate(x, size=(H//4, W//4), mode='bilinear', align_corners=True) for x in features]
        features = [conv(feature) for conv, feature in zip(self.decode_head, features)]

        features = torch.cat(features, dim=1)
        depth = self.final_conv(features)
        depth = F.interpolate(depth, size=(H, W), mode='bilinear', align_corners=True)

        return depth

    def _reshape_as_image(self, x):
        B, N, C = x.shape
        H = W = int(N ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, C, H, W)
        return x

    @classmethod
    def from_pretrained(cls, model_path):
        model = cls()
        state_dict = torch.hub.load_state_dict_from_url(
            'https://huggingface.co/LiheYoung/depth_anything_vitl14/resolve/main/depth_anything_vitl14.pth',
            map_location='cpu'
        )
        model.load_state_dict(state_dict)
        return model