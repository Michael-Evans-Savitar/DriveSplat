#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
import os
import numpy as np
import time

"""
GPU selection policy:
- Respect CUDA_VISIBLE_DEVICES when it is set by the caller.
- If it is unset, choose the GPU with the lowest reported memory usage.
"""
if os.environ.get("CUDA_VISIBLE_DEVICES", "").strip() == "":
    try:
        import subprocess
        result = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        used = [int(x.strip()) for x in result.splitlines() if x.strip()]
        if used:
            os.environ['CUDA_VISIBLE_DEVICES'] = str(np.argmin(used))
    except Exception:
        pass

print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}")


import torch
import json
import wandb
import time
from os import makedirs
import shutil
from pathlib import Path
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import network_gui
import sys
from scene import Scene, DriveSplatModel, Dataset
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams, ModelHiddenParams
from utils.timer import Timer
import torch.nn.functional as F
from gaussian_renderer.prefilter_voxel import PrefilterVoxel
from gaussian_renderer.drivesplat_render import DriveSplatRenderer
from utils.image_utils import save_img_torch, visualize_depth_numpy
from torch.cuda.amp import autocast
from scene.deform_model import DeformModel

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
    print("found tf board")
except ImportError:
    TENSORBOARD_FOUND = False
    print("not found tf board")


time_record = {
    "total_training_time_minutes": 0.0,
    "total_testing_time_minutes": 0.0,
    "total_time_minutes": 0.0
}

def saveRuntimeCode(dst: str) -> None:
    additionalIgnorePatterns = ['.git', '.gitignore']
    ignorePatterns = set()
    ROOT = '.'
    with open(os.path.join(ROOT, '.gitignore')) as gitIgnoreFile:
        for line in gitIgnoreFile:
            if not line.startswith('#'):
                if line.endswith('\n'):
                    line = line[:-1]
                if line.endswith('/'):
                    line = line[:-1]
                ignorePatterns.add(line)
    ignorePatterns = list(ignorePatterns)
    for additionalPattern in additionalIgnorePatterns:
        ignorePatterns.append(additionalPattern)

    log_dir = Path(__file__).resolve().parent

    shutil.copytree(log_dir, dst, ignore=shutil.ignore_patterns(*ignorePatterns))

    print('Backup Finished!')


def scene_reconstruction(dataset, opt, hyper, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, gaussians, scene, stage, train_iter, timer, wandb=None, logger=None, args=None):
    first_iter = 0
    final_iter = train_iter
    tb_writer = prepare_output_and_logger(dataset)
    gaussians.training_setup(opt)
    deform = DeformModel(is_blender=False, is_6dof=False)
    deform.train_setting(opt)
    gaussians.set_coarse_interval(opt.coarse_iter, opt.coarse_factor)
    if checkpoint:
        if stage == "coarse" and stage not in checkpoint:
            print("start from fine stage, skip coarse stage.")
            return
        if stage in checkpoint:
            (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)
    elif args is not None and getattr(args, "resume_iteration", 0) > 0:
        first_iter = int(args.resume_iteration)
        logger.info(f"[Resume] Continuing from saved model iteration {first_iter}; optimizer state is reinitialized.")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    gaussian_render = DriveSplatRenderer(pipe, hyper)
    prefilter = PrefilterVoxel(pipe)

    viewpoint_stack = None
    ema_loss_for_log = 0.0



    import gc
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[Memory]  Cleaned up initialization memory before training")

    progress_bar = tqdm(range(first_iter, final_iter), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, final_iter + 1):
        gaussians.update_learning_rate(iteration)

        if dataset.random_background:
            bg_color = [np.random.random(),np.random.random(),np.random.random()]
        elif dataset.white_background:
            bg_color = [1.0, 1.0, 1.0]
        else:
            bg_color = [0.0, 0.0, 0.0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        # Pick a random camera.
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        total_frames = len(viewpoint_stack)
        viewpoint_cam = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render the current training view.
        if (iteration - 1) == debug_from:
            pipe.debug = True


        # Lightweight profiling window.
        do_profile = (iteration >= 1000 and iteration <= 1200 and iteration % 10 == 0)

        if do_profile:
            torch.cuda.synchronize()
            t_start = time.time()

        gaussians.set_visibility(include_list=list(set(gaussians.model_name_id.keys())))

        if do_profile:
            torch.cuda.synchronize()
            t1 = time.time()
            print(f"\n[Iter {iteration}] set_visibility: {(t1-t_start)*1000:.2f}ms")

        gaussians.parse_camera(viewpoint_cam)

        if do_profile:
            torch.cuda.synchronize()
            t2 = time.time()
            print(f"[Iter {iteration}] parse_camera: {(t2-t1)*1000:.2f}ms")

        if do_profile:
            torch.cuda.synchronize()
            t3 = time.time()

        gaussians.set_anchor_mask(viewpoint_cam.camera_center, iteration, viewpoint_cam.resolution_scale)

        if do_profile:
            torch.cuda.synchronize()
            t4 = time.time()
            print(f"[Iter {iteration}] set_anchor_mask: {(t4-t3)*1000:.2f}ms")

        voxel_visible_mask = prefilter.prefilter_voxel(viewpoint_cam, gaussians, pipe, background)

        if do_profile:
            torch.cuda.synchronize()
            t5 = time.time()
            print(f"[Iter {iteration}] prefilter_voxel: {(t5-t4)*1000:.2f}ms")

        retain_grad = (iteration < opt.update_until and iteration >= 0)



        if do_profile:
            torch.cuda.synchronize()
            t_render_start = time.time()

        render_pkg = gaussian_render.render(viewpoint_cam, gaussians, pipe, hyper, background, visible_mask=voxel_visible_mask, retain_grad=retain_grad, stage=stage, total_frames=total_frames, iteration=iteration, deform=deform, is_infer=False)

        if do_profile:
            torch.cuda.synchronize()
            t_render_end = time.time()
            print(f"[Iter {iteration}]  RENDER: {(t_render_end-t_render_start)*1000:.2f}ms <- ")
        image, acc, viewspace_point_tensor, visibility_filter, offset_selection_mask, radii, scaling, opacity = (
            render_pkg["render"], render_pkg["acc"], render_pkg["viewspace_points"], render_pkg["visibility_filter"],
            render_pkg["selection_mask"], render_pkg["radii"], render_pkg["scaling"], render_pkg["neural_opacity"])
        rendered_depth = render_pkg["rendered_depth"]
        total_ids = render_pkg["total_ids"]
        opacity_ids = render_pkg["neural_opacity_ids"]
        rendered_normal = render_pkg['normals']
        if hyper.use_semantic:
            semantic = render_pkg['semantic']

        has_nan = torch.isnan(image).any() or torch.isnan(rendered_depth).any() or torch.isnan(rendered_normal).any()

        visible_count = visibility_filter.sum().item() if visibility_filter is not None else 0

        if has_nan:
            print(f"[ERROR] Rendered output contains NaN at iteration {iteration}")
            print(f"  - image has NaN: {torch.isnan(image).any().item()}")
            print(f"  - rendered_depth has NaN: {torch.isnan(rendered_depth).any().item()}")
            print(f"  - rendered_normal has NaN: {torch.isnan(rendered_normal).any().item()}")
            print(f"  - visible gaussians: {visible_count}")
            loss = torch.tensor(0.0, device=image.device, requires_grad=True)
            rgb_loss = loss
            tb_writer.add_scalar("fine/rgb_loss", 0.0, iteration)
        elif visible_count == 0:
            print(f"[ERROR] No visible gaussians at iteration {iteration}, skipping this iteration")

            if hasattr(gaussians, '_anchor_mask'):
                total_anchors = gaussians._anchor_mask.shape[0] if gaussians._anchor_mask is not None else 0
                selected_anchors = gaussians._anchor_mask.sum().item() if gaussians._anchor_mask is not None else 0
                print(f"  - Total anchors: {total_anchors}, Selected anchors: {selected_anchors}")
            if hasattr(gaussians, 'background') and hasattr(gaussians.background, '_anchor_mask'):
                bg_anchors = gaussians.background._anchor_mask.sum().item() if gaussians.background._anchor_mask is not None else 0
                print(f"  - Background anchors: {bg_anchors}")
            if hasattr(gaussians, 'graph_obj_list'):
                for obj_name in gaussians.graph_obj_list:
                    obj_model = getattr(gaussians, obj_name, None)
                    if obj_model is not None and hasattr(obj_model, '_anchor_mask'):
                        obj_anchors = obj_model._anchor_mask.sum().item() if obj_model._anchor_mask is not None else 0
                        print(f"  - {obj_name} anchors: {obj_anchors}")
            if voxel_visible_mask is not None:
                voxel_selected = voxel_visible_mask.sum().item()
                print(f"  - Voxel visible mask: {voxel_selected} selected")
            loss = torch.tensor(0.0, device=image.device, requires_grad=True)
            rgb_loss = loss
            tb_writer.add_scalar("fine/rgb_loss", 0.0, iteration)
        else:
            gt_image = viewpoint_cam.original_image.cuda()
            Ll1 = l1_loss(image, gt_image)

            ssim_loss = (1.0 - ssim(image, gt_image))

            if torch.isnan(Ll1) or torch.isnan(ssim_loss):
                print(f"[ERROR] RGB loss is NaN at iteration {iteration}")
                print(f"  - Ll1 is NaN: {torch.isnan(Ll1).item()}")
                print(f"  - ssim_loss is NaN: {torch.isnan(ssim_loss).item()}")
                print(f"  - visible gaussians: {visible_count}")
                loss = torch.tensor(0.0, device=image.device, requires_grad=True)
                rgb_loss = loss
                tb_writer.add_scalar("fine/rgb_loss", 0.0, iteration)
            else:
                loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * ssim_loss
                rgb_loss = loss

                if rgb_loss.item() > 0.5 and iteration % 100 == 0:
                    print(f"[WARNING] High loss at iteration {iteration}: {rgb_loss.item():.6f}")
                    print(f"  - Ll1: {Ll1.item():.6f}, ssim_loss: {ssim_loss.item():.6f}")
                    print(f"  - visible gaussians: {visible_count}")
                    print(f"  - image mean: {image.mean().item():.6f}, std: {image.std().item():.6f}")
                    print(f"  - gt_image mean: {gt_image.mean().item():.6f}, std: {gt_image.std().item():.6f}")

                tb_writer.add_scalar("fine/rgb_loss", rgb_loss.item(), iteration)

        if hasattr(viewpoint_cam, 'original_obj_bound'):
            obj_bound = viewpoint_cam.original_obj_bound.cuda().bool()
        else:
            obj_bound = torch.zeros_like(gt_image[0:1]).bool()

        if hasattr(viewpoint_cam, 'original_sky_mask'):
            sky_mask = viewpoint_cam.original_sky_mask.cuda()
        else:
            sky_mask = None

        if hasattr(viewpoint_cam, 'original_mask'):
            mask = viewpoint_cam.original_mask.cuda().bool()
        else:
            mask = torch.ones_like(gt_image[0:1]).bool()

        if opt.lambda_depth > 0 and stage == "fine" and iteration > 100 and not has_nan:
            gt_depth = viewpoint_cam.depth.cuda()
            from torchmetrics.functional.regression import pearson_corrcoef
            gt_depth = gt_depth.squeeze(0)
            rendered_depth_new = rendered_depth.squeeze(0)

            if torch.isnan(rendered_depth_new).any():
                print(f"[WARNING] Rendered depth contains NaN at iteration {iteration}, skipping depth loss")
            elif torch.isnan(gt_depth).any():
                print(f"[WARNING] GT depth contains NaN at iteration {iteration}, skipping depth loss")
            elif sky_mask is not None:
                valid_mask = ~sky_mask.squeeze()
                gt_depth_valid = gt_depth[valid_mask]
                rendered_depth_valid = rendered_depth_new[valid_mask]


                if gt_depth_valid.numel() > 0 and rendered_depth_valid.numel() > 0:
                    depth_mode = "relative"
                    if depth_mode == "metric":
                        depth_loss = F.l1_loss(gt_depth_valid, rendered_depth_valid)
                        depth_loss = opt.lambda_depth * depth_loss
                        tb_writer.add_scalar("depth_loss", depth_loss.item(), iteration)
                        loss = loss + depth_loss
                    else:
                        gt_std = gt_depth_valid.std()
                        rendered_std = rendered_depth_valid.std()
                        if gt_std > 1e-6 and rendered_std > 1e-6:
                            # Relative depth supervision via inverse-depth Pearson correlation.
                            pearson = pearson_corrcoef(1.0 / (gt_depth_valid + 200.), rendered_depth_valid)


                            if isinstance(pearson, torch.Tensor):
                                if pearson.numel() > 1:
                                    pearson = pearson.mean()
                            else:
                                pearson = torch.tensor(float(pearson), device=rendered_depth_valid.device)

                            if torch.isnan(pearson):
                                pearson = torch.tensor(0.0, device=rendered_depth_valid.device)

                            depth_loss = 1.0 - pearson
                            loss_depth = opt.lambda_depth * depth_loss
                            tb_writer.add_scalar("depth_loss", loss_depth.item(), iteration)
                            loss = loss + loss_depth
            elif not torch.isnan(rendered_depth_new).any() and not torch.isnan(gt_depth).any():
                    depth_mode = "relative"
                    if depth_mode == "metric":
                        depth_loss = F.l1_loss(gt_depth, rendered_depth_new)
                        depth_loss = opt.lambda_depth * depth_loss
                        tb_writer.add_scalar("depth_loss", depth_loss.item(), iteration)
                        loss = loss + depth_loss
                    else:
                        gt_std = gt_depth.std()
                        rendered_std = rendered_depth_new.std()
                        if gt_std > 1e-6 and rendered_std > 1e-6:
                            pearson = pearson_corrcoef(1 / (gt_depth + 200.), rendered_depth_new)


                            if isinstance(pearson, torch.Tensor):
                                if pearson.numel() > 1:
                                    pearson = pearson.mean()
                            else:
                                pearson = torch.tensor(float(pearson), device=rendered_depth_new.device)

                            if torch.isnan(pearson):
                                pearson = torch.tensor(0.0, device=rendered_depth_new.device)

                            depth_loss = 1.0 - pearson

                            if torch.isnan(depth_loss):
                                depth_loss = torch.tensor(0.0, device=rendered_depth_new.device)

                            loss_depth = opt.lambda_depth * depth_loss
                            tb_writer.add_scalar("depth_loss", loss_depth.item(), iteration)
                            loss = loss + loss_depth

        if opt.lambda_reg > 0 and gaussians.include_obj and stage == "fine" and iteration >= opt.update_until:
            gaussians.set_obj_anchor_mask(viewpoint_cam.camera_center, iteration, viewpoint_cam.resolution_scale)
            # Render objects without gradients for the regularization target.
            with torch.no_grad():
                render_pkg_obj = gaussian_render.render_object(viewpoint_cam, gaussians, pipe, hyper, background, visible_mask=voxel_visible_mask, total_frames=total_frames, iteration=iteration, deform=deform, is_infer=True)
            image_obj, acc_obj = render_pkg_obj["render"], render_pkg_obj['acc']
            acc_obj = torch.clamp(acc_obj, min=1e-6, max=1.-1e-6)
            obj_acc_loss = torch.where(obj_bound,
                -(acc_obj * torch.log(acc_obj) +  (1. - acc_obj) * torch.log(1. - acc_obj)),
                -torch.log(1. - acc_obj)).mean()
            tb_writer.add_scalar("reg_loss", obj_acc_loss.item(), iteration)
            loss = loss + opt.lambda_reg * obj_acc_loss

        # Sky loss.
        if opt.lambda_sky > 0 and gaussians.include_sky and sky_mask is not None:
            acc = torch.clamp(acc, min=1e-6, max=1.-1e-6)
            sky_loss = torch.where(sky_mask, -torch.log(1 - acc), -torch.log(acc)).mean()
            if len(opt.lambda_sky_scale) > 0:
                sky_loss *= opt.lambda_sky_scale[viewpoint_cam.meta['cam']]
            loss = loss + opt.lambda_sky * sky_loss

        if opt.lambda_normal > 0 and stage == "fine" and iteration > 100 and not has_nan:
                gt_normal = viewpoint_cam.normal_gt.cuda()
                gt_normal = gt_normal.permute(1, 2, 0)
                gt_normal = gt_normal * 2.0 - 1.0
                R_c2w = viewpoint_cam.world_view_transform[:3, :3]
                gt_normal = torch.matmul(gt_normal, R_c2w.T)
                normal_pred = rendered_normal.permute(1, 2, 0)

                if torch.isnan(gt_normal).any() or torch.isnan(normal_pred).any():
                    print(f"[WARNING] GT normal or rendered normal contains NaN at iteration {iteration}, skipping normal loss")
                    print(f"  - gt_normal has NaN: {torch.isnan(gt_normal).any().item()}")
                    print(f"  - normal_pred has NaN: {torch.isnan(normal_pred).any().item()}")
                else:
                    normal_l1_loss = torch.abs(normal_pred - gt_normal).mean()
                    normal_cos_loss = (1. - torch.sum(normal_pred * gt_normal, dim=-1)).mean()

                    if torch.isnan(normal_l1_loss) or torch.isnan(normal_cos_loss):
                        print(f"[WARNING] Normal loss is NaN at iteration {iteration}, skipping")
                        print(f"  - normal_l1_loss is NaN: {torch.isnan(normal_l1_loss).item()}")
                        print(f"  - normal_cos_loss is NaN: {torch.isnan(normal_cos_loss).item()}")
                    else:
                        loss_normal = normal_l1_loss + normal_cos_loss
                        loss_normal = opt.lambda_normal * loss_normal
                        if torch.isnan(loss_normal):
                            print(f"[WARNING] Final normal loss is NaN at iteration {iteration}, skipping")
                        else:
                            tb_writer.add_scalar("normal_loss", loss_normal.item(), iteration)
                            tb_writer.add_scalar("normal_l1_loss", normal_l1_loss*opt.lambda_normal)
                            tb_writer.add_scalar("normal_cos_loss", normal_cos_loss*opt.lambda_normal)
                            loss = loss + loss_normal

        # Color correction loss.
        if opt.lambda_color_correction > 0 and gaussians.use_color_correction:
            color_correction_reg_loss = gaussians.color_correction.regularization_loss(viewpoint_cam)
            tb_writer.add_scalar('color_correction_reg_loss',  color_correction_reg_loss.item(), iteration)
            loss = loss + opt.lambda_color_correction * color_correction_reg_loss

        # Pose correction loss.
        if opt.lambda_pose_correction > 0 and gaussians.use_pose_correction:
            pose_correction_reg_loss = gaussians.pose_correction.regularization_loss()
            tb_writer.add_scalar('pose_correction_reg_loss',  pose_correction_reg_loss.item(), iteration)
            loss = loss + opt.lambda_pose_correction * pose_correction_reg_loss

        # LiDAR depth loss.
        if stage == "fine" and iteration > 100 and opt.lambda_lidar_depth > 0:
            gt_lidar_depth = viewpoint_cam.lidar_depth.cuda()
            gt_lidar_depth = gt_lidar_depth.squeeze(0)
            lidar_mask = gt_lidar_depth > 0
            gt_depth = gt_lidar_depth[lidar_mask]
            rendered_lidar_depth = rendered_depth.squeeze(0)
            pred_depth = rendered_lidar_depth[lidar_mask]
            loss_lidar_depth = opt.lambda_lidar_depth * F.l1_loss(gt_depth, pred_depth)
            tb_writer.add_scalar("lidar_depth_loss", loss_lidar_depth.item(), iteration)
            loss = loss + loss_lidar_depth

        # Appearance embedding regularization.
        if hasattr(opt, 'lambda_appearance_reg') and opt.lambda_appearance_reg > 0 and gaussians.appearance_dim > 0:
            appearance_reg_loss = 0.0
            for model_name in gaussians.model_name_id.keys():
                model = getattr(gaussians, model_name)
                if hasattr(model, 'embedding_appearance') and model.embedding_appearance is not None:
                    appearance_reg_loss += torch.mean(model.embedding_appearance.weight ** 2)
            if appearance_reg_loss > 0:
                appearance_reg_loss = opt.lambda_appearance_reg * appearance_reg_loss
                tb_writer.add_scalar("appearance_reg_loss", appearance_reg_loss.item(), iteration)
                loss = loss + appearance_reg_loss

        tb_writer.add_scalar("total loss", loss.item(), iteration)

        if do_profile:
            torch.cuda.synchronize()
            t_backward_start = time.time()

        loss.backward()

        if iteration % 5 == 0:
            torch.cuda.empty_cache()

        if do_profile:
            torch.cuda.synchronize()
            t_backward_end = time.time()
            print(f"[Iter {iteration}] backward: {(t_backward_end-t_backward_start)*1000:.2f}ms")

            total_time = t_backward_end - t_start
            print(f"[Iter {iteration}]   TOTAL: {total_time*1000:.2f}ms ({1.0/total_time:.2f} it/s)\n")

        iter_end.record()


        is_save_images = True
        if is_save_images and iteration % 1000 == 0:
            if sky_mask is not None:
                depth_show = rendered_depth * (~sky_mask).float()
                gt_depth_show = viewpoint_cam.depth.cuda() * (~sky_mask).float()
            else:
                depth_show = rendered_depth
                gt_depth_show = viewpoint_cam.depth.cuda().float()
            depth_colored, _ = visualize_depth_numpy(depth_show.detach().cpu().numpy().squeeze(0))
            depth_colored = depth_colored[..., [2, 1, 0]] / 255
            depth_colored = torch.from_numpy(depth_colored).permute(2, 0, 1).float().cuda()

            gt_depth_colored, _ = visualize_depth_numpy(gt_depth_show.detach().cpu().numpy().squeeze(0))
            gt_depth_colored = gt_depth_colored[..., [2, 1, 0]] / 255
            gt_depth_colored = torch.from_numpy(gt_depth_colored).permute(2, 0, 1).float().cuda()
            if sky_mask is not None:
                rendered_normal = torch.clamp(rendered_normal * (~sky_mask).float(), -1.0, 1.0).detach().cpu().numpy()
            else:
                rendered_normal = torch.clamp(rendered_normal, -1.0, 1.0).detach().cpu().numpy()
            rendered_normal = np.array(rendered_normal)
            normal_colored = (rendered_normal + 1.0) / 2.0
            normal_colored = torch.from_numpy(normal_colored).float().cuda()
            gt_normal = torch.clamp(viewpoint_cam.normal_gt.cuda(), 0, 1.0).detach().cpu().numpy()
            gt_normal = np.array(gt_normal)
            gt_normal_colored = gt_normal
            gt_normal_colored = torch.from_numpy(gt_normal_colored).float().cuda()
            acc = acc.repeat(3, 1, 1)
            if gaussians.include_obj:
                with torch.no_grad():
                    render_pkg_obj = gaussian_render.render_object(
                        viewpoint_cam,
                        gaussians,
                        pipe,
                        hyper,
                        visible_mask=voxel_visible_mask,
                        total_frames=total_frames,
                        iteration=iteration,
                        deform=deform,
                        is_infer=False,
                    )
                    image_obj, acc_obj = render_pkg_obj["render"], render_pkg_obj["acc"]
                acc_obj = acc_obj.repeat(3, 1, 1)
            obj_bound = obj_bound.repeat(3, 1, 1)
            row0 = torch.cat([gt_image, obj_bound, gt_depth_colored], dim=2)
            if gaussians.include_obj:
                row1 = torch.cat([image, image_obj, depth_colored], dim=2)
            else:
                row1 = torch.cat([image, depth_colored, normal_colored], dim=2)
            image_to_show = torch.cat([row0, row1], dim=1)
            image_to_show = torch.clamp(image_to_show, 0.0, 1.0)
            os.makedirs(f"{dataset.model_path}/log_images", exist_ok=True)
            save_img_torch(image_to_show, f"{dataset.model_path}/log_images/{iteration}.png")

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log

            if iteration % 10 == 0:
                progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # densification
            if iteration < opt.update_until and iteration > opt.start_stat:
                gaussians.training_statis(viewspace_point_tensor, opacity, visibility_filter, offset_selection_mask, voxel_visible_mask, total_ids, opacity_ids)

                if opt.update_anchor and iteration > opt.update_from and iteration % opt.update_interval == 0:
                    non_rigid_start_iter = getattr(hyper, 'non_rigid_start_iter', 3000) if hyper is not None else 3000
                    is_non_rigid_active = iteration >= non_rigid_start_iter

                    if is_non_rigid_active:
                        # Densify background and dynamic actors after non-rigid training starts.
                        if hasattr(gaussians, 'background'):
                            gaussians.background.adjust_anchor(
                                iteration=iteration,
                                check_interval=opt.update_interval,
                                success_threshold=opt.success_threshold,
                                grad_threshold=opt.densify_grad_threshold_bkgd,
                                update_ratio=dataset.update_ratio,
                                extra_ratio=dataset.extra_ratio,
                                extra_up=dataset.extra_up,
                                min_opacity=opt.min_opacity
                            )

                        if hasattr(gaussians, 'graph_gaussian_range'):
                            for model_name in gaussians.graph_gaussian_range.keys():
                                if model_name == 'background':
                                    continue
                                model = getattr(gaussians, model_name)
                                if not hasattr(model, 'adjust_anchor'):
                                    continue

                                is_deformable = bool(getattr(model, 'deformable', False))
                                is_pedestrian = (getattr(model, 'obj_class', None) == 'pedestrian')
                                is_nonrigid_obj = is_deformable or is_pedestrian

                                grad_threshold = opt.densify_grad_threshold_bkgd
                                update_ratio = dataset.update_ratio
                                if is_nonrigid_obj:
                                    grad_threshold = opt.densify_grad_threshold_bkgd * 2.5
                                    update_ratio = dataset.update_ratio * 0.7

                                model.adjust_anchor(
                                    iteration=iteration,
                                    check_interval=opt.update_interval,
                                    success_threshold=opt.success_threshold,
                                    grad_threshold=grad_threshold,
                                    update_ratio=update_ratio,
                                    extra_ratio=dataset.extra_ratio,
                                    extra_up=dataset.extra_up,
                                    min_opacity=opt.min_opacity,
                                    viewpoint_camera=getattr(gaussians, 'viewpoint_camera', None)
                                )
                    else:
                        gaussians.adjust_anchor(
                            iteration=iteration,
                            check_interval=opt.update_interval,
                            success_threshold=opt.success_threshold,
                            grad_threshold=opt.densify_grad_threshold_bkgd,
                            update_ratio=dataset.update_ratio,
                            extra_ratio=dataset.extra_ratio,
                            extra_up=dataset.extra_up,
                            min_opacity=opt.min_opacity
                        )
            elif iteration == opt.update_until:
                del gaussians.opacity_accum
                del gaussians.offset_gradient_accum
                del gaussians.offset_denom
                torch.cuda.empty_cache()
            if iteration in testing_iterations:
                print(f"model name {gaussians.model_name_id}")
                training_report(gaussians, tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, testing_iterations, scene, gaussian_render.render, pipe, hyper, background, wandb, logger, stage, deform=deform)
            if (iteration in saving_iterations):
                logger.info("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration, stage)

            # Optimizer step
            if iteration < opt.iterations:
                gaussians.update_optimizer(iteration, stage=stage)
                # Actor-specific optimizers, including pedestrian AGD, are stepped inside
                # DriveSplatModel.update_optimizer() -> GaussianModelActor.update_optimizer().

            if iteration % 20 == 0:
                torch.cuda.empty_cache()

            if (iteration in checkpoint_iterations):
                logger.info("\n[ITER {}] Saving Checkpoint".format(iteration))
                model_params = gaussians.capture()
                ckpt_dir = os.path.join(scene.model_path, "chkpnt")
                os.makedirs(ckpt_dir, exist_ok=True)
                ckpt_path = os.path.join(ckpt_dir, f"{stage}_{iteration}.pth")
                torch.save((model_params, iteration), ckpt_path)
                logger.info(f"[ITER {iteration}] Checkpoint saved to: {ckpt_path}")

def training(dataset, hyper, opt, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, wandb=None, logger=None, ply_path=None, args=None):
    logger.info("Start training...")

    data = Dataset(dataset, hyper)
    gaussians = DriveSplatModel(data.scene_info.metadata, dataset.sh_degree, hyper,
        dataset.feat_dim, dataset.n_offsets, dataset.fork, dataset.use_feat_bank, dataset.appearance_dim,
        dataset.add_opacity_dist, dataset.add_cov_dist, dataset.add_color_dist, dataset.add_level,
        dataset.visible_threshold, dataset.dist2level, dataset.base_layer, dataset.progressive, dataset.extend
    )
    timer = Timer()
    resume_iteration = int(getattr(args, "resume_iteration", 0) or 0) if args is not None else 0
    load_iteration = resume_iteration if resume_iteration > 0 else None
    scene = Scene(dataset, data, gaussians, load_iteration=load_iteration, ply_path=ply_path, shuffle=False, logger=logger, resolution_scales=dataset.resolution_scales)
    timer.start()
    scene_reconstruction(dataset, opt, hyper, pipe, dataset_name, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, gaussians, scene, "fine", opt.iterations, timer, wandb, logger, args=args)

    return scene

def prepare_output_and_logger(args):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])

    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(gaussians, tb_writer, dataset_name, iteration, Ll1, loss, l1_loss, testing_iterations, scene : Scene, renderFunc, pipe, hyper, bg, wandb=None, logger=None, stage="fine", deform=None):
    if tb_writer:
        tb_writer.add_scalar(f"{dataset_name}/train_loss_patches/l1_loss", Ll1.item(), iteration)
        tb_writer.add_scalar(f"{dataset_name}/train_loss_patches/total_loss", loss.item(), iteration)

    if wandb is not None:
        wandb.log({"train_l1_loss": Ll1, "train_total_loss": loss})


def get_logger(path):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    fileinfo = logging.FileHandler(os.path.join(path, "outputs.log"))
    fileinfo.setLevel(logging.INFO)
    controlshow = logging.StreamHandler()
    controlshow.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
    fileinfo.setFormatter(formatter)
    controlshow.setFormatter(formatter)

    logger.addHandler(fileinfo)
    logger.addHandler(controlshow)

    return logger


if __name__ == "__main__":

    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    hp = ModelHiddenParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6005)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--warmup', action='store_true', default=False)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[-1])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[-1])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--resume_iteration", type=int, default=0, help="Resume training from point_cloud/iteration_N model state without optimizer state.")
    parser.add_argument("--gpu", type=str, default = '-1')
    parser.add_argument("--configs", type=str, default = "")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    if args.configs:
        import mmcv
        from utils.params_utils import merge_hparams
        config = mmcv.Config.fromfile(args.configs)
        args = merge_hparams(args, config)
    print("Optimizing " + args.model_path)

    # Enable logging.
    model_path = args.model_path
    os.makedirs(model_path, exist_ok=True)

    logger = get_logger(model_path)

    logger.info(f'args: {args}')

    if args.test_iterations[0] == -1:
        args.test_iterations = [i for i in range(5000, args.iterations + 1, 5000)]
    if len(args.test_iterations) == 0 or args.test_iterations[-1] != args.iterations:
        args.test_iterations.append(args.iterations)
    print(args.test_iterations)

    if args.save_iterations[0] == -1:
        args.save_iterations = [i for i in range(5000, args.iterations + 1, 5000)]
    if len(args.save_iterations) == 0 or args.save_iterations[-1] != args.iterations:
        args.save_iterations.append(args.iterations)
    print(args.save_iterations)

    if args.gpu != '-1':
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
        logger.info(f'using GPU {args.gpu}')

    dataset = args.source_path.split('/')[-1]
    exp_name = args.model_path.split('/')[-2]

    if args.use_wandb:
        wandb.login()
        run = wandb.init(
            # Set the project where this run will be logged
            project=f"DriveSplat-{dataset}",
            name=exp_name,
            # Track hyperparameters and run metadata
            settings=wandb.Settings(start_method="fork"),
            config=vars(args)
        )
    else:
        wandb = None

    logger.info("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)

    # Train the scene.
    scene = training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb, logger, args=args)
    if args.warmup:
        logger.info("\n Warmup finished! Reboot from last checkpoints")
        new_ply_path = os.path.join(args.model_path, f'point_cloud/iteration_{args.iterations}', 'point_cloud.ply')
        scene = training(lp.extract(args), hp.extract(args), op.extract(args), pp.extract(args), dataset,  args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, wandb=wandb, logger=logger, ply_path=new_ply_path, args=args)

    # All done
    logger.info("\nTraining complete.")


    # Save a compact snapshot used by render.py.
    full_snapshot_path = os.path.join(args.model_path, f"point_cloud/iteration_{args.iterations}", "full_model_snapshot.pth")
    try:
        full_snapshot = {
            'iteration': args.iterations,
            'model_names': list(scene.gaussians.model_name_id.keys()),
        }
        for model_name in scene.gaussians.model_name_id.keys():
            model = getattr(scene.gaussians, model_name)
            try:
                full_snapshot[f'{model_name}_voxel_size'] = getattr(model, 'voxel_size', None)
                full_snapshot[f'{model_name}_standard_dist'] = getattr(model, 'standard_dist', None)
                full_snapshot[f'{model_name}_levels'] = getattr(model, 'levels', None)
                bounds = getattr(model, 'bounds', None)
                if bounds is not None and hasattr(bounds, 'cpu'):
                    full_snapshot[f'{model_name}_bounds'] = bounds.cpu()
                else:
                    full_snapshot[f'{model_name}_bounds'] = bounds
                main_dir = getattr(model, 'main_direction', None)
                if main_dir is not None and hasattr(main_dir, 'cpu'):
                    full_snapshot[f'{model_name}_main_direction'] = main_dir.cpu()
                else:
                    full_snapshot[f'{model_name}_main_direction'] = main_dir
                init_pos = getattr(model, 'init_pos', None)
                if init_pos is not None and hasattr(init_pos, 'cpu'):
                    full_snapshot[f'{model_name}_init_pos'] = init_pos.cpu()
                else:
                    full_snapshot[f'{model_name}_init_pos'] = init_pos
            except Exception as e:
                logger.warning(f" Failed to save {model_name} basic attrs: {e}")
            try:
                if hasattr(model, 'hash_encoding') and model.hash_encoding is not None:
                    full_snapshot[f'{model_name}_hash_encoding_state'] = model.hash_encoding.state_dict()
            except Exception as e:
                logger.warning(f" Failed to save {model_name} hash_encoding: {e}")
            try:
                if hasattr(model, 'hash_lod_partitioner') and model.hash_lod_partitioner is not None:
                    full_snapshot[f'{model_name}_hash_lod_partitioner_state'] = model.hash_lod_partitioner.save_state()
            except Exception as e:
                logger.warning(f" Failed to save {model_name} hash_lod_partitioner: {e}")
        torch.save(full_snapshot, full_snapshot_path)
        logger.info(f" Saved full model snapshot to {full_snapshot_path}")
    except Exception as e:
        logger.warning(f" Failed to save full snapshot: {e}")
        import traceback
        traceback.print_exc()
