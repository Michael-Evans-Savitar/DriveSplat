#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
import os
from os import makedirs
import torch
import numpy as np
import open3d as o3d

def set_default_cuda_device():
    if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        return
    try:
        import subprocess
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        used = [int(x.strip()) for x in result.splitlines() if x.strip()]
        if used:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(np.argmin(used))
    except Exception:
        pass


set_default_cuda_device()
print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")

from scene import Scene, DriveSplatModel, Dataset
import json
import time
from gaussian_renderer import render, prefilter_voxel
import torchvision
from tqdm import tqdm
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer.prefilter_voxel import PrefilterVoxel
from gaussian_renderer.drivesplat_render import DriveSplatRenderer
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from utils.image_utils import save_img_torch, visualize_depth_numpy, visualize_normal_numpy
from utils.timer import Timer
import wandb
import sys
import cv2
import matplotlib.pyplot as plt
from utils.system_utils import makedirs
from scipy.spatial import cKDTree
from skimage import measure
import imageio
import plotly.graph_objects as go
from scipy.spatial.transform import Rotation
from scene.deform_model import DeformModel

def render_set(model_path, name, iteration, views, gaussians, pipeline, hyper, opt, background, deform, is_infer=False):
    prefilter = PrefilterVoxel(pipeline, background)
    gaussian_render = DriveSplatRenderer(pipeline, hyper)


    if len(views) > 0:
        print(f"[render.py render_set] render parameters ({name} set, iter={iteration}):")
        print(f"  - voxel_size: {gaussians.voxel_size}")
        print(f"  - standard_dist: {gaussians.standard_dist}")
        if hasattr(gaussians.background, 'hash_encoding') and gaussians.background.hash_encoding is not None:
            he = gaussians.background.hash_encoding
            print(f"  - hash_encoding.training: {he.training}")
            # hash encoding
            for pname, param in he.named_parameters():
                print(f"  - hash_encoding.{pname}: mean={param.mean().item():.6f}, std={param.std().item():.6f}")
                break

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    render_obj_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_obj")
    gt_mask_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_mask")
    error_path = os.path.join(model_path, name, "ours_{}".format(iteration), "errors")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "depths")
    gt_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_depth")
    gt_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_normal")
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normals")
    semantic_path = os.path.join(model_path, name, "ours_{}".format(iteration), "semantics")
    obj_path = os.path.join(model_path, name, "ours_{}".format(iteration), "obj")
    mono_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "mono_depth")
    video_path = os.path.join(model_path, name, "ours_{}".format(iteration), "videos")
    makedirs(render_path, exist_ok=True)
    makedirs(render_obj_path, exist_ok=True)
    makedirs(error_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(depth_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)
    makedirs(gt_depth_path, exist_ok=True)
    makedirs(gt_normal_path, exist_ok=True)
    makedirs(semantic_path, exist_ok=True)
    makedirs(obj_path, exist_ok=True)
    makedirs(mono_depth_path, exist_ok=True)
    makedirs(video_path, exist_ok=True)
    makedirs(gt_mask_path, exist_ok=True)

    t_list = []
    visible_count_list = []
    per_view_dict = {}
    rgb_frames = []
    depth_frames = []
    normal_frames = []
    total_frames = len(views)
    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        gaussians.set_visibility(include_list=list(set(gaussians.model_name_id.keys())))
        gaussians.parse_camera(view)
        gaussians.set_anchor_mask(view.camera_center, iteration, view.resolution_scale)
        voxel_visible_mask = prefilter.prefilter_voxel(view, gaussians, pipeline, background)
        t_start = time.time()
        render_pkg = gaussian_render.render(view, gaussians, pipeline, hyper, background, visible_mask=voxel_visible_mask, total_frames=total_frames, iteration=iteration, deform=deform, is_infer=is_infer)
        t_end = time.time()

        t_list.append(t_end - t_start)

        # renders
        rendering = torch.clamp(render_pkg["render"], 0.0, 1.0)
        visible_count = render_pkg["visibility_filter"].sum()
        visible_count_list.append(visible_count)
        # gts
        gt = view.original_image[0:3, :, :]
        gt_mask = view.original_mask.cuda().bool()
        gt_depth = view.lidar_depth
        gt_normal = view.normal_gt
        # error maps
        if gt.device != rendering.device:
            rendering = rendering.to(gt.device)
        errormap = (rendering - gt).abs()
        sky_mask = view.original_sky_mask.cuda()
        rendered_depth = render_pkg["rendered_depth"]
        if sky_mask is not None:
            depth_show = 10 / rendered_depth * (~sky_mask).float()
            gt_depth_show = gt_depth.cuda() * (~sky_mask).float()
            mono_depth_show = view.depth.cuda() * (~sky_mask).float()
        else:
            depth_show = rendered_depth
            gt_depth_show = gt_depth.cuda().float()
            mono_depth_show = view.depth.cuda().float()
        rendered_depth, _ = visualize_depth_numpy(depth_show.detach().cpu().numpy().squeeze(0))
        rendered_depth = rendered_depth[..., [2, 1, 0]] / 255
        rendered_depth = torch.from_numpy(rendered_depth).permute(2, 0, 1).float().cuda()
        gt_depth, _ = visualize_depth_numpy(gt_depth_show.detach().cpu().numpy().squeeze(0))
        gt_depth = gt_depth[..., [2, 1, 0]] / 255
        gt_depth = torch.from_numpy(gt_depth).permute(2, 0, 1).float().cuda()
        mono_depth, _ = visualize_depth_numpy(mono_depth_show.detach().cpu().numpy().squeeze(0))
        mono_depth = mono_depth[..., [2, 1, 0]] / 255
        mono_depth = torch.from_numpy(mono_depth).permute(2, 0, 1).float().cuda()
        rendered_normal = render_pkg["normals"]
        rendered_normal = rendered_normal * (~sky_mask)
        rendered_normal = (rendered_normal + 1.0) / 2.0
        gt_normal = torch.clamp(gt_normal.cuda(), 0, 1.0).detach().cpu()
        depth_save = render_pkg["rendered_depth"].squeeze(0).detach().cpu().numpy()
        gt_depth_save = view.lidar_depth.squeeze(0).detach().cpu().numpy()
        mono_depth_save = view.depth.squeeze(0).detach().cpu().numpy()
        normal_save = render_pkg["normals"].detach().cpu().numpy()
        gt_normal_save = view.normal_gt.detach().cpu().numpy()
        np.save(os.path.join(depth_path, '{0:05d}'.format(idx) + '.npy'), depth_save)
        np.save(os.path.join(gt_depth_path, '{0:05d}'.format(idx) + '.npy'), gt_depth_save)
        np.save(os.path.join(normal_path, '{0:05d}'.format(idx) + '.npy'), normal_save)
        np.save(os.path.join(gt_normal_path, '{0:05d}'.format(idx) + '.npy'), gt_normal_save)
        np.save(os.path.join(mono_depth_path, '{0:05d}'.format(idx) + '.npy'), mono_depth_save)
        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(errormap, os.path.join(error_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt_mask.float(), os.path.join(gt_mask_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(rendered_depth, os.path.join(depth_path, '{0:05d}'.format(idx) + '.tiff'))
        torchvision.utils.save_image(rendered_normal, os.path.join(normal_path, '{0:05d}'.format(idx) + ".tiff"))
        torchvision.utils.save_image(gt_depth, os.path.join(gt_depth_path, '{0:05d}'.format(idx) + '.tiff'))
        torchvision.utils.save_image(gt_normal, os.path.join(gt_normal_path, '{0:05d}'.format(idx) + ".tiff"))

        per_view_dict['{0:05d}'.format(idx) + ".png"] = visible_count.item()


        rgb_frames.append((rendering * 255).byte().permute(1,2,0).cpu().numpy())
        depth_frames.append((rendered_depth * 255).byte().permute(1,2,0).cpu().numpy())
        normal_frames.append((rendered_normal * 255).byte().permute(1,2,0).cpu().numpy())

    with open(os.path.join(model_path, name, "ours_{}".format(iteration), "per_view_count.json"), 'w') as fp:
            json.dump(per_view_dict, fp, indent=True)


    def save_video(frames, path, fps=30):
        h, w = frames[0].shape[:2]
        writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        writer.release()

    # RGB
    save_video(rgb_frames, os.path.join(video_path, 'rgb.mp4'))


    save_video(depth_frames, os.path.join(video_path, 'depth.mp4'))


    save_video(normal_frames, os.path.join(video_path, 'normal.mp4'))

    return t_list, visible_count_list

def render_sets(dataset, iteration, hyper, pipeline, opt, skip_train=False, skip_test=False, wandb=None, tb_writer=None, dataset_name=None, logger=None):
    with torch.no_grad():
        data = Dataset(dataset, hyper)
        gaussians = DriveSplatModel(data.scene_info.metadata, dataset.sh_degree, hyper,
            dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank, dataset.appearance_dim,
            dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.add_level,
            dataset.visible_threshold, dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend
        )
        timer = Timer()
        scene = Scene(dataset, data, gaussians, load_iteration=iteration, shuffle=False, logger=logger, resolution_scales=dataset.resolution_scales)
        deform = DeformModel(is_blender=False, is_6dof=False)
        gaussians.eval()


        full_snapshot_path = os.path.join(
            dataset.model_path,
            f"point_cloud/iteration_{scene.loaded_iter}",
            "full_model_snapshot.pth"
        )
        if os.path.exists(full_snapshot_path):
            try:
                full_snapshot = torch.load(full_snapshot_path, map_location="cuda")
                print(f" Loaded full model snapshot from {full_snapshot_path}")


                if 'voxel_size' in full_snapshot:
                    gaussians.voxel_size = full_snapshot['voxel_size']
                if 'standard_dist' in full_snapshot:
                    gaussians.standard_dist = full_snapshot['standard_dist']
                if 'levels' in full_snapshot:
                    gaussians.levels = full_snapshot['levels']


                for model_name in gaussians.model_name_id.keys():
                    model = getattr(gaussians, model_name)
                    if f'{model_name}_voxel_size' in full_snapshot and full_snapshot[f'{model_name}_voxel_size'] is not None:
                        model.voxel_size = full_snapshot[f'{model_name}_voxel_size']
                    if f'{model_name}_standard_dist' in full_snapshot and full_snapshot[f'{model_name}_standard_dist'] is not None:
                        model.standard_dist = full_snapshot[f'{model_name}_standard_dist']
                    if f'{model_name}_levels' in full_snapshot and full_snapshot[f'{model_name}_levels'] is not None:
                        model.levels = full_snapshot[f'{model_name}_levels']
                    if f'{model_name}_bounds' in full_snapshot and full_snapshot[f'{model_name}_bounds'] is not None:
                        model.bounds = full_snapshot[f'{model_name}_bounds']
                    if f'{model_name}_main_direction' in full_snapshot and full_snapshot[f'{model_name}_main_direction'] is not None:
                        model.main_direction = full_snapshot[f'{model_name}_main_direction']
                    if f'{model_name}_init_pos' in full_snapshot and full_snapshot[f'{model_name}_init_pos'] is not None:
                        model.init_pos = full_snapshot[f'{model_name}_init_pos']
                    #   hash_encoding ()
                    if f'{model_name}_hash_encoding_state' in full_snapshot:
                        if hasattr(model, 'hash_encoding') and model.hash_encoding is not None:
                            model.hash_encoding.load_state_dict(full_snapshot[f'{model_name}_hash_encoding_state'])
                            print(f"   Restored {model_name} hash_encoding from snapshot")
                    #   hash_lod_partitioner ()
                    if f'{model_name}_hash_lod_partitioner_state' in full_snapshot:
                        if hasattr(model, 'hash_lod_partitioner') and model.hash_lod_partitioner is not None:
                            model.hash_lod_partitioner.load_state(full_snapshot[f'{model_name}_hash_lod_partitioner_state'])
                            print(f"   Restored {model_name} hash_lod_partitioner from snapshot")

                print(f" Full model snapshot restored successfully")
            except Exception as e:
                print(f" Failed to load full snapshot: {e}")
        else:
            print(f" Full model snapshot not found at {full_snapshot_path}, using standard loading")


        ellipsoid_dir = os.path.join(dataset.model_path, "ellipsoid_views")
        makedirs(ellipsoid_dir, exist_ok=True)

        if dataset.random_background:
            bg_color = [np.random.random(),np.random.random(),np.random.random()]
        elif dataset.white_background:
            bg_color = [1.0, 1.0, 1.0]
        else:
            bg_color = [0.0, 0.0, 0.0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
        if not os.path.exists(dataset.model_path):
            os.makedirs(dataset.model_path)
        if not skip_train:
            t_train_list, visible_count  = render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, hyper, opt, background, deform)
            train_fps = 1.0 / torch.tensor(t_train_list[5:]).mean()
            print(f"train_fps: {train_fps}")

        if not skip_test:
            t_test_list, visible_count = render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, hyper, opt, background, deform, is_infer=True)
            test_fps = 1.0 / torch.tensor(t_test_list[5:]).mean()
            if tb_writer:
                tb_writer.add_scalar(f'{dataset_name}/test_FPS', test_fps.item(), 0)
            print(f"test_fps: {test_fps}")

        return visible_count


def save_level_points(background_model):
    """level,,levelscaling"""
    import numpy as np
    import os
    import open3d as o3d


    points = background_model.get_anchor
    levels = background_model.get_level
    scalings = background_model.get_scaling  # scaling

    # numpy
    points_np = points.detach().cpu().numpy()   # shape: (N, 3)
    levels_np = levels.detach().cpu().numpy().squeeze()   # shape: (N,),
    scalings_np = scalings.detach().cpu().numpy()  # shape: (N, 6)


    print(f"Points shape: {points_np.shape}")
    print(f"Levels shape: {levels_np.shape}")
    print(f"Scalings shape: {scalings_np.shape}")


    save_dir = os.path.join(os.getcwd(), "level_points")
    os.makedirs(save_dir, exist_ok=True)

    # level
    unique_levels = np.unique(levels_np)
    print(f"Unique levels: {unique_levels}")

    # level
    level_stats = {}

    # level
    for level in unique_levels:
        # level
        mask = (levels_np == level)
        level_points = points_np[mask]  # shape: (M, 3)
        level_scalings = scalings_np[mask]  # shape: (M, 6)


        print(f"\nProcessing level {level}:")
        print(f"  - point count: {np.sum(mask)}")
        print(f"  - point shape: {level_points.shape}")
        print(f"  - scaling shape: {level_scalings.shape}")

        # level
        avg_scaling = np.mean(level_scalings, axis=0)
        point_count = len(level_points)

        # Open3D
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(level_points)

        # scaling
        colors = np.zeros((point_count, 3))
        scaling_magnitude = np.mean(level_scalings[:, :3], axis=1)
        if point_count > 0:
            normalized_scaling = (scaling_magnitude - np.min(scaling_magnitude)) / (np.max(scaling_magnitude) - np.min(scaling_magnitude) + 1e-6)
            colors[:, 0] = normalized_scaling  # scaling
        pcd.colors = o3d.utility.Vector3dVector(colors)

        # ply
        save_path_ply = os.path.join(save_dir, f"level_{level}_points.ply")
        o3d.io.write_point_cloud(save_path_ply, pcd)

        # npy
        save_path_npy = os.path.join(save_dir, f"level_{level}_data.npz")
        np.savez(save_path_npy,
                 points=level_points,
                 scalings=level_scalings)


        level_stats[int(level)] = {
            "point_count": point_count,
            "average_scaling": avg_scaling.tolist(),
            "min_scaling": np.min(level_scalings, axis=0).tolist() if point_count > 0 else None,
            "max_scaling": np.max(level_scalings, axis=0).tolist() if point_count > 0 else None
        }

        print(f"Level {level}:")
        print(f"  - point count: {point_count}")
        print(f"  - average scaling: {avg_scaling}")
        print(f"  - point cloud path: {save_path_ply}")

    # JSON
    import json
    stats_path = os.path.join(save_dir, "level_statistics.json")
    with open(stats_path, "w") as f:
        json.dump(level_stats, f, indent=2)

    print(f"Level statistics path: {stats_path}")

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--ape", default=10, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--show_level", action="store_true")
    parser.add_argument('--port', type=int, default=6005)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[-1])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[-1])
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--gpu", type=str, default = '-1')
    parser.add_argument("--configs", type=str, default = "")
    args = parser.parse_args(sys.argv[1:])
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)

    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    visible_count = render_sets(lp.extract(args), -1, hp.extract(args), pp.extract(args), op.extract(args), skip_train=args.skip_train, skip_test=args.skip_test, wandb=wandb)
