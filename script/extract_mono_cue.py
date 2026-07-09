"""
@file   extract_mono_cues.py
@brief  extract monocular cues (normal & depth)
        Adapted from https://github.com/EPFL-VILAB/omnidata

Installation:
    git clone https://github.com/EPFL-VILAB/omnidata

    pip install einops joblib pandas h5py scipy seaborn kornia timm pytorch-lightning
"""

import os
import sys
import argparse
import imageio
import numpy as np
from glob import glob
from tqdm import tqdm

import PIL
import skimage
from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torchvision import transforms
import torchvision.transforms.functional as transF

def list_contains(l: list, v):
    """
    Whether any item in `l` contains `v`
    """
    for item in l:
        if v in item:
            return True
    else:
        return False

def standardize_depth_map(img, mask_valid=None, trunc_value=0.1):
    if mask_valid is not None:
        img[~mask_valid] = torch.nan
    sorted_img = torch.sort(torch.flatten(img))[0]
    # Remove nan, nan at the end of sort
    num_nan = sorted_img.isnan().sum()
    if num_nan > 0:
        sorted_img = sorted_img[:-num_nan]
    # Remove outliers
    trunc_img = sorted_img[int(trunc_value * len(sorted_img)): int((1 - trunc_value) * len(sorted_img))]
    trunc_mean = trunc_img.mean()
    trunc_var = trunc_img.var()
    eps = 1e-6
    # Replace nan by mean
    img = torch.nan_to_num(img, nan=trunc_mean)
    # Standardize
    img = (img - trunc_mean) / torch.sqrt(trunc_var + eps)
    return img

def extract_cues(img_path: str, output_path_base: str, ref_img_size: int=384, verbose=True):
    with torch.no_grad():
        img = imageio.imread(img_path)
        img = skimage.img_as_float32(img)
        H, W, _ = img.shape
        img_tensor = trans_totensor(img)[:3].unsqueeze(0).to(device)
        if H < W:
            H_ = ref_img_size
            W_ = int(((W / (H/H_)) // 32)) * 32 # Force to be a multiple of 32
        else:
            W_ = ref_img_size
            H_ = int(((H / (W/W_)) // 32)) * 32 # Force to be a multiple of 32
        img_tensor = transF.resize(img_tensor, (H_, W_), antialias=True)

        if img_tensor.shape[1] == 1:
            img_tensor = img_tensor.repeat_interleave(3,1)

        output = model(img_tensor).clamp(min=0, max=1)

        if args.task == 'depth':
            output = output.clamp(0,1).data / output.max()
            output = transF.resize(output, (H,W), antialias=True).movedim(0,-1).cpu().numpy()
            # plt.imsave(f"{output_path_base}.png", output, cmap='viridis')
            verbose = True
            if verbose:
                imageio.imwrite(f"{output_path_base}.jpg", (output*255).clip(0,255).astype(np.uint8)[..., 0], format="jpg") # Fastest and smallest file size
            # NOTE: jianfei: Although saving to float16 is lossy, we are allowing it since it's just extracting some weak hint here.
            np.savez_compressed(f"{output_path_base}.npz", output.astype(np.float16))

        else:
            original_output = output.data.clamp(0,1).squeeze(0)
            # Resize to original shape
            # NOTE: jianfei: Although saving to uint8 is lossy, we are allowing it since it's just extracting some weak hint here.
            output = transF.resize(original_output, (H,W), antialias=True).movedim(0,-1).mul_(255.).to(torch.uint8).clamp_(0,255).cpu().numpy()
            output_numpy = transF.resize(original_output, (H,W), antialias=True).movedim(0,-1).cpu().numpy()
            print(f"output content: {output_numpy}")
            np.save(f"{output_path_base}.npy", output_numpy)
            # plt.imsave(f"{output_path_base}.png", output/2+0.5)
            imageio.imwrite(f"{output_path_base}.jpg", output) # Fastest and smallest file size
            # imageio.imwrite(f"{output_path_base}.png", output) # Very slow

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Visualize output for depth or surface normals')

    # Dataset specific configs
    parser.add_argument('--data_root', type=str, default='./data/waymo_scenes')
    parser.add_argument('--seq_list', type=str, default=['2011_09_26_drive_0017_sync'], help='specify --seq_list if you want to limit the list of seqs')
    parser.add_argument('--verbose', action='store_true', help="Additionally generate .jpg files for visualization")
    parser.add_argument('--ignore_existing', action='store_true')
    parser.add_argument('--rgb_dirname', type=str, default="images")
    parser.add_argument('--depth_dirname', type=str, default="depth")
    parser.add_argument('--normals_dirname', type=str, default="normal")

    # Algorithm configs
    parser.add_argument('--task', dest='task', required=True, default=None, help="normal or depth")
    parser.add_argument('--omnidata_path', dest='omnidata_path', help="path to omnidata model",
                        default='omnidata/omnidata_tools/torch')
    parser.add_argument('--pretrained_models', dest='pretrained_models', help="path to pretrained models",
                        default='omnidata/omnidata_tools/torch/pretrained_models')
    parser.add_argument('--ref_img_size', dest='ref_img_size', type=int, default=512,
                        help="image size when inference (will still save full-scale output)")
    args = parser.parse_args()


    #-- Original preparation

    if args.pretrained_models is None:
        args.pretrained_models = os.path.join(args.omnidata_path, "pretrained_models")

    sys.path.append(args.omnidata_path)
    print(sys.path)
    from omnidata.omnidata_tools.torch.modules.unet import UNet
    from omnidata.omnidata_tools.torch.modules.midas.dpt_depth import DPTDepthModel
    from omnidata.omnidata_tools.torch.data.transforms import get_transform

    trans_topil = transforms.ToPILImage()
    map_location = (lambda storage, loc: storage.cuda()) if torch.cuda.is_available() else torch.device('cpu')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # get target task and model
    if args.task == 'normal' or args.task == 'normals':
        image_size = 384

        #---- Version 1 model
        #         state_dict[k.replace('model.', '')] = v


        pretrained_weights_path = os.path.join(args.pretrained_models, 'omnidata_dpt_normal_v2.ckpt')
        model = DPTDepthModel(backbone='vitb_rn50_384', num_channels=3) # DPT Hybrid
        checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint

        model.load_state_dict(state_dict)
        model.to(device)
        #                                     transforms.CenterCrop(image_size),

        trans_totensor = transforms.Compose([
            get_transform('rgb', image_size=None)])

    elif args.task == 'depth':
        image_size = 384
        pretrained_weights_path = os.path.join(args.pretrained_models, 'omnidata_dpt_depth_v2.ckpt')  # 'omnidata_dpt_depth_v1.ckpt'
        model = DPTDepthModel(backbone='vitb_rn50_384') # DPT Hybrid
        checkpoint = torch.load(pretrained_weights_path, map_location=map_location)
        if 'state_dict' in checkpoint:
            state_dict = {}
            for k, v in checkpoint['state_dict'].items():
                state_dict[k[6:]] = v
        else:
            state_dict = checkpoint
        model.load_state_dict(state_dict)
        model.to(device)
        #                                     transforms.CenterCrop(image_size),

        #                                     transforms.Normalize(mean=0.5, std=0.5)])
        trans_totensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=0.5, std=0.5)])

    else:
        print("task should be one of the following: normal, depth")
        sys.exit()


    #--- Dataset Specific processing

    if isinstance(args.seq_list, list):
        select_scene_ids = args.seq_list
    else:
        select_scene_ids = [args.seq_list]

    for scene_i, scene_id in enumerate(tqdm(select_scene_ids, f'Extracting {args.task} ...')):
        for i in range(1):
            image_dir = os.path.join(args.data_root, scene_id, f"{args.rgb_dirname}")
            obs_id_list = sorted(os.listdir(image_dir))

            if args.task == 'depth':
                output_dir = image_dir.replace(args.rgb_dirname, args.depth_dirname)
            elif args.task == 'normal':
                output_dir = image_dir.replace(args.rgb_dirname, args.normals_dirname)
            else:
                raise RuntimeError(f"Invalid task={args.task}")

            os.makedirs(output_dir, exist_ok=True)

            for obs_i, obs_id in enumerate(tqdm(obs_id_list, f'scene [{scene_i}/{len(select_scene_ids)}]')):
                img_path = os.path.join(image_dir, obs_id)
                fbase = os.path.splitext(os.path.basename(img_path))[0]

                if args.task == 'depth':
                    output_base = os.path.join(output_dir, fbase)
                elif args.task == 'normal':
                    output_base = os.path.join(output_dir, fbase)
                else:
                    raise RuntimeError(f"Invalid task={args.task}")

                if args.ignore_existing and list_contains(os.listdir(output_dir), fbase):
                    continue

                #---- Inference and save outputs
                extract_cues(img_path, output_base, args.ref_img_size, verbose=args.verbose)