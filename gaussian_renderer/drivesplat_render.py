#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
import torch
from einops import repeat
from utils.sh_utils import eval_sh
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model_all import DriveSplatModel
from utils.general_utils import quaternion_to_matrix
from gaussian_renderer.generate_neural_gaussian import GenerateNeuralGaussians
import time

class DriveSplatRenderer:
    def __init__(self, pipe, hyper):
        self.pipe = pipe
        self.hyper = hyper
    def render_all(self, viewpoint_camera, pc : DriveSplatModel, pipe, hyper, bg_color : torch.Tensor, scaling_modifier=1.0, visible_mask=None, stage = "fine", retain_grad=False, ape_code=-1, override_color=None, total_frames=50, iteration=0, deform=None, is_infer=False):
        """(,)"""
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()


        result = {}



        result = self.render(viewpoint_camera, pc, pipe, hyper, bg_color, scaling_modifier, visible_mask, stage, retain_grad, ape_code, override_color, total_frames, iteration=iteration, deform=deform, is_infer=is_infer)


        import threading
        import queue

        render_results = queue.Queue()


        def render_bg_thread():

            current_visibility = {name: pc.get_visibility(name) for name in pc.graph_gaussian_range.keys()}

            pc.set_visibility(include_list=['background'])

            bg_result = self.render_kernel(viewpoint_camera, pc, pipe, hyper, bg_color, scaling_modifier, visible_mask, stage, retain_grad, ape_code, override_color, total_frames, iteration=iteration, deform=deform, is_infer=is_infer)

            for name, vis in current_visibility.items():
                pc.set_visibility(name, vis)

            render_results.put(('background', bg_result))


        def render_obj_thread():

            current_visibility = {name: pc.get_visibility(name) for name in pc.graph_gaussian_range.keys()}

            pc.set_visibility(include_list=pc.obj_list)

            obj_result = self.render_kernel(viewpoint_camera, pc, pipe, hyper, bg_color, scaling_modifier, visible_mask, stage, retain_grad, ape_code, override_color, total_frames, iteration=iteration, deform=deform, is_infer=is_infer)

            for name, vis in current_visibility.items():
                pc.set_visibility(name, vis)

            render_results.put(('object', obj_result))


        bg_thread = threading.Thread(target=render_bg_thread)
        obj_thread = threading.Thread(target=render_obj_thread)

        bg_thread.start()
        obj_thread.start()


        bg_thread.join()
        obj_thread.join()


        while not render_results.empty():
            render_type, render_data = render_results.get()
            if render_type == 'background':
                result['rgb_background'] = render_data['rgb'] if 'rgb' in render_data else render_data['render']
                result['acc_background'] = render_data['acc']
            elif render_type == 'object':
                result['rgb_object'] = render_data['rgb'] if 'rgb' in render_data else render_data['render']
                result['acc_object'] = render_data['acc']

        end.record()
        torch.cuda.synchronize()
        elapsed_time = start.elapsed_time(end)

        return result

    def render_object(self, viewpoint_camera, pc : DriveSplatModel, pipe, hyper, bg_color =[1, 1, 1], scaling_modifier=1.0, visible_mask=None, stage = "fine", retain_grad=False, ape_code=-1, override_color=None, total_frames=50, iteration=0, deform=None, is_infer=False):

        original_include_list = getattr(pc, 'include_list', None)
        original_graph_obj_list = pc.graph_obj_list.copy() if hasattr(pc, 'graph_obj_list') and pc.graph_obj_list is not None else None
        original_graph_gaussian_range = pc.graph_gaussian_range.copy() if hasattr(pc, 'graph_gaussian_range') and pc.graph_gaussian_range is not None else None

        pc.set_visibility(include_list=pc.obj_list)
        bg_color = [1, 1, 1]
        bg_color = torch.tensor(bg_color).float().cuda()
        generate_neural = GenerateNeuralGaussians(pipe)
        is_training = pc.get_color_mlp.training

        total_gaussians = 0
        for rng in pc.graph_gaussian_range.values():
            total_gaussians = max(total_gaussians, rng[1] + 1)

        total_xyz = None
        total_color = None
        total_opacity = None
        total_scaling = None
        total_rot = None
        total_normals = None
        total_semantics = None
        total_ids = None
        filled_mask = None

        all_neural_opacity, all_mask, neural_opacity_ids = [], [], []
        current_start = 0
        for model_idx, model_name in enumerate(pc.graph_gaussian_range.keys()):
            start, end = pc.graph_gaussian_range[model_name]
            end = end + 1
            model = getattr(pc, model_name)
            single_mask = model.visible_mask if hasattr(model, 'visible_mask') and model.visible_mask is not None else (visible_mask[start:end] if visible_mask is not None else None)
            if not hasattr(model, 'model_name'):
                model.model_name = model_name
            if is_training:
                if model_name == 'background':
                    continue
                elif model.obj_class == 'pedestrian':
                    xyz, color, opacity, scaling, rot, neural_opacity, mask, normals, semantics, shs_features = generate_neural.generate_neural_gaussians_non_rigid(
                        pipe, viewpoint_camera, model, single_mask, is_training=is_training, stage=stage, total_frames=total_frames, iteration=iteration, deform=deform, is_infer=is_infer, hyper=hyper)
                else:
                    xyz, color, opacity, scaling, rot, neural_opacity, mask, normals, semantics, shs_features = generate_neural.generate_neural_gaussians_obj(
                        pipe, viewpoint_camera, model, single_mask, is_training=is_training, stage=stage)
                all_neural_opacity.append(neural_opacity)
                all_mask.append(mask)
                neural_opacity_ids.append(
                    torch.full((neural_opacity.shape[0],), model_idx, dtype=torch.int32, device="cuda"))
            else:
                if model_name == 'background':
                    continue
                elif model.obj_class == 'pedestrian':
                    xyz, color, opacity, scaling, rot, normals, semantics, shs_features = generate_neural.generate_neural_gaussians_non_rigid(
                        pipe, viewpoint_camera, model, single_mask, is_training=is_training, ape_code=ape_code,
                        stage=stage, total_frames=total_frames, iteration=iteration, deform=deform, is_infer=is_infer, hyper=hyper)
                else:
                    xyz, color, opacity, scaling, rot, normals, semantics, shs_features = generate_neural.generate_neural_gaussians_obj(
                        pipe, viewpoint_camera, model, single_mask, is_training=is_training, ape_code=ape_code,
                        stage=stage)

            n = xyz.shape[0]


            if total_xyz is None:
                device = xyz.device
                dtype = xyz.dtype


                actual_total = max(total_gaussians, current_start + n)
                total_xyz = torch.empty((actual_total, xyz.shape[1]), device=device, dtype=dtype)
                total_color = torch.empty((actual_total, color.shape[1]), device=device, dtype=color.dtype)
                total_opacity = torch.empty((actual_total, opacity.shape[1]), device=device, dtype=opacity.dtype)
                total_scaling = torch.empty((actual_total, scaling.shape[1]), device=device, dtype=scaling.dtype)
                total_rot = torch.empty((actual_total, rot.shape[1]), device=device, dtype=rot.dtype)
                total_normals = torch.empty((actual_total, normals.shape[1]), device=device, dtype=normals.dtype)
                total_semantics = torch.empty((actual_total, semantics.shape[1]), device=device, dtype=semantics.dtype)
                total_ids = torch.empty((actual_total,), device=device, dtype=torch.int32)
                filled_mask = torch.zeros((actual_total,), device=device, dtype=torch.bool)
            else:

                if current_start + n > total_xyz.shape[0]:
                    new_size = max(total_xyz.shape[0] * 2, current_start + n)
                    total_xyz = torch.cat([total_xyz, torch.empty((new_size - total_xyz.shape[0], total_xyz.shape[1]), device=total_xyz.device, dtype=total_xyz.dtype)], dim=0)
                    total_color = torch.cat([total_color, torch.empty((new_size - total_color.shape[0], total_color.shape[1]), device=total_color.device, dtype=total_color.dtype)], dim=0)
                    total_opacity = torch.cat([total_opacity, torch.empty((new_size - total_opacity.shape[0], total_opacity.shape[1]), device=total_opacity.device, dtype=total_opacity.dtype)], dim=0)
                    total_scaling = torch.cat([total_scaling, torch.empty((new_size - total_scaling.shape[0], total_scaling.shape[1]), device=total_scaling.device, dtype=total_scaling.dtype)], dim=0)
                    total_rot = torch.cat([total_rot, torch.empty((new_size - total_rot.shape[0], total_rot.shape[1]), device=total_rot.device, dtype=total_rot.dtype)], dim=0)
                    total_normals = torch.cat([total_normals, torch.empty((new_size - total_normals.shape[0], total_normals.shape[1]), device=total_normals.device, dtype=total_normals.dtype)], dim=0)
                    total_semantics = torch.cat([total_semantics, torch.empty((new_size - total_semantics.shape[0], total_semantics.shape[1]), device=total_semantics.device, dtype=total_semantics.dtype)], dim=0)
                    total_ids = torch.cat([total_ids, torch.empty((new_size - total_ids.shape[0],), device=total_ids.device, dtype=total_ids.dtype)], dim=0)
                    filled_mask = torch.cat([filled_mask, torch.zeros((new_size - filled_mask.shape[0],), device=filled_mask.device, dtype=filled_mask.dtype)], dim=0)

            total_xyz[current_start:current_start+n] = xyz
            total_color[current_start:current_start+n] = color
            total_opacity[current_start:current_start+n] = opacity
            total_scaling[current_start:current_start+n] = scaling
            total_rot[current_start:current_start+n] = rot
            total_normals[current_start:current_start+n] = normals
            total_semantics[current_start:current_start+n] = semantics
            total_ids[current_start:current_start+n] = model_idx
            filled_mask[current_start:current_start+n] = True


            current_start += n


        if is_training:
            def concat_all_features(feature_list, feature_name):
                if not feature_list:

                    return None
                try:
                    concatenated = torch.cat(feature_list, dim=0)
                    if any(tensor.requires_grad for tensor in feature_list):
                        concatenated.requires_grad_(True)
                    return concatenated
                except Exception as e:
                    print(f"Error concatenating {feature_name}: {e}")
                    return None
            total_neural_opacity = concat_all_features(all_neural_opacity, "all_neural_opacity")
            total_mask = concat_all_features(all_mask, "all_mask")

        if total_xyz is None:

            if original_include_list is not None:
                pc.include_list = original_include_list
            if original_graph_obj_list is not None:
                pc.graph_obj_list = original_graph_obj_list
            if original_graph_gaussian_range is not None:
                pc.graph_gaussian_range = original_graph_gaussian_range


            H, W = int(viewpoint_camera.image_height), int(viewpoint_camera.image_width)
            empty_image = torch.zeros((3, H, W), device="cuda")
            empty_acc = torch.zeros((1, H, W), device="cuda")
            empty_depth = torch.zeros((1, H, W), device="cuda")
            empty_screenspace = torch.zeros((0, 3), device="cuda", requires_grad=True)

            result = {
                "render": empty_image,
                "acc": empty_acc,
                "rendered_depth": empty_depth,
                "viewspace_points": empty_screenspace,
                "visibility_filter": torch.zeros((0,), device="cuda", dtype=torch.bool),
                "radii": torch.zeros((0,), device="cuda", dtype=torch.int32),
            }
            if is_training:
                result.update({
                    "selection_mask": None,
                    "neural_opacity": None,
                    "scaling": None,
                    "total_ids": None,
                    "neural_opacity_ids": [],
                    "total_xyz": None,
                })
            else:
                result.update({
                    "ids": None,
                    "neural_opacity_ids": [],
                })
            return result


        if filled_mask is not None:
            total_xyz = total_xyz[filled_mask]
            total_color = total_color[filled_mask]
            total_opacity = total_opacity[filled_mask]
            total_scaling = total_scaling[filled_mask]
            total_rot = total_rot[filled_mask]
            total_normals = total_normals[filled_mask]
            total_semantics = total_semantics[filled_mask]
            total_ids = total_ids[filled_mask]

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(total_xyz, dtype=pc._anchor.dtype, requires_grad=True,
                                              device="cuda") + 0

        if retain_grad:
            try:
                screenspace_points.retain_grad()
            except:
                pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=pc.active_sh_degree,  # 1, pc.active_sh_degree
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        feature_names = []
        feature_dims = []
        features = []

        if pipe.render_normal:
            feature_names.append('normals')
            feature_dims.append(total_normals.shape[-1])
            features.append(total_normals)

        if hyper.use_semantic:
            feature_names.append('semantic')
            feature_dims.append(total_semantics.shape[-1])
            features.append(total_semantics)

        if len(features) > 0:
            features = torch.cat(features, dim=-1)
        else:
            features = None

        # color + random color
        # offset grad = 0
        # anchor grad != 0
        # anchor,
        # xyzloss
        rendered_image, radii, depth, rendered_acc, rendered_feature = rasterizer(
            means3D=total_xyz,  # total_xyz
            means2D=screenspace_points,
            shs=None,
            colors_precomp=total_color,  # total_color,
            opacities=total_opacity,  # total_opacity
            scales=total_scaling,  # total_scaling
            rotations=total_rot,  # total_rot
            cov3D_precomp=None,
            semantics=features)

        rendered_feature_dict = dict()
        if rendered_feature.shape[0] > 0:
            rendered_feature_list = torch.split(rendered_feature, feature_dims, dim=0)
            for i, feature_name in enumerate(feature_names):
                rendered_feature_dict[feature_name] = rendered_feature_list[i]
        if 'normals' in rendered_feature_dict:
            rendered_feature_dict['normals'] = torch.nn.functional.normalize(rendered_feature_dict['normals'], dim=0)
        if 'semantic' in rendered_feature_dict:
            rendered_semantic = rendered_feature_dict['semantic']
            semantic_mode = hyper.semantic_mode
            if semantic_mode == 'logits':
                pass
            else:
                rendered_semantic = rendered_semantic / (
                            torch.sum(rendered_semantic, dim=0, keepdim=True) + 1e-8)  # normalize to probabilities
                rendered_semantic = torch.log(rendered_semantic + 1e-8)  # change for cross entropy loss

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        if is_training:
            result = {"render": rendered_image,
                      "acc": rendered_acc,
                      "rendered_depth": depth,
                      "viewspace_points": screenspace_points,
                      "visibility_filter": radii > 0,
                      "radii": radii,
                      "selection_mask": total_mask,
                      "neural_opacity": total_neural_opacity,
                      "scaling": total_scaling,
                      "total_ids": total_ids,
                      "neural_opacity_ids": neural_opacity_ids,
                      "total_xyz": total_xyz,
                      }
        else:
            result = {"render": rendered_image,
                      "acc": rendered_acc,
                      "rendered_depth": depth,
                      "viewspace_points": screenspace_points,
                      "visibility_filter": radii > 0,
                      "radii": radii,
                      "ids": total_ids,
                      "neural_opacity_ids": neural_opacity_ids,
                      }
        result.update(rendered_feature_dict)


        if original_include_list is not None:
            pc.include_list = original_include_list
        if original_graph_obj_list is not None:
            pc.graph_obj_list = original_graph_obj_list
        if original_graph_gaussian_range is not None:
            pc.graph_gaussian_range = original_graph_gaussian_range

        return result

    def render_background(self, viewpoint_camera, pc : DriveSplatModel, pipe, hyper, bg_color : torch.Tensor, scaling_modifier=1.0, visible_mask=None, stage = "fine", retain_grad=False, ape_code=-1, override_color=None, total_frames=50):
        pc.set_visibility(include_list=['background'])
        pc.parse_camera(viewpoint_camera)
        result = self.render_kernel(viewpoint_camera, pc, pipe, hyper, bg_color, scaling_modifier, visible_mask, stage, retain_grad, ape_code, override_color, total_frames)

        return result

    def render_sky(
        self,
        viewpoint_camera,
        pc,
        convert_SHs_python = None,
        compute_cov3D_python = None,
        scaling_modifier = None,
        override_color = None
    ):
        pc.set_visibility(include_list=['sky'])
        pc.parse_camera(viewpoint_camera)
        result = self.render_kernel(viewpoint_camera, pc, convert_SHs_python, compute_cov3D_python, scaling_modifier, override_color)
        return result

    def render(self, viewpoint_camera, pc : DriveSplatModel, pipe, hyper, bg_color : torch.Tensor, scaling_modifier=1.0, visible_mask=None, stage = "fine", retain_grad=False, ape_code=-1, override_color=None, total_frames=50, iteration=0, deform=None, is_infer=False):

        result = self.render_kernel(viewpoint_camera, pc, pipe, hyper, bg_color, scaling_modifier, visible_mask, stage, retain_grad, ape_code, override_color, total_frames, iteration=iteration, deform=deform, is_infer=is_infer)

        # Step2: render sky
        if getattr(pc, 'include_sky', False) and hasattr(pc, 'sky_cubemap'):
            sky_color = pc.sky_cubemap(viewpoint_camera, result['acc'].detach(), hyper)

            result['render'] = result['render'] + sky_color * (1 - result['acc'])

        if pc.use_color_correction:
            result['render'] = pc.color_correction(viewpoint_camera, result['render'])

        return result


    def render_kernel(self, viewpoint_camera, pc : DriveSplatModel, pipe, hyper, bg_color : torch.Tensor, scaling_modifier=1.0, visible_mask=None, stage = "fine", retain_grad=False, ape_code=-1, override_color=None, total_frames=50, iteration=0, deform=None, is_infer=False):
        """
        Render the scene.

        Background tensor (bg_color) must be on GPU!
        """

        generate_neural = GenerateNeuralGaussians(pipe)
        is_training = pc.get_color_mlp.training

        all_xyz, all_color, all_opacity, all_scaling = [], [], [], []
        all_rot, all_normals, all_semantics, all_ids = [], [], [], []
        all_neural_opacity, all_mask, neural_opacity_ids = [], [], []
        start_time = time.time()
        for model_idx, model_name in enumerate(pc.graph_gaussian_range.keys()):
            start, end = pc.graph_gaussian_range[model_name]
            end = end + 1
            model = getattr(pc, model_name)
            single_mask = model.visible_mask
            if is_training:
                if model_name == 'background':
                    xyz, color, opacity, scaling, rot, neural_opacity, mask, normals, semantics, shs_features = generate_neural.generate_neural_gaussians_kernel(
                        pipe, viewpoint_camera, model, single_mask, is_training=is_training, stage=stage)
                elif model.obj_class == 'pedestrian':
                    xyz, color, opacity, scaling, rot, neural_opacity, mask, normals, semantics, shs_features = generate_neural.generate_neural_gaussians_non_rigid(
                        pipe, viewpoint_camera, model, single_mask, is_training=is_training, stage=stage, total_frames=total_frames, iteration=iteration, deform=deform, is_infer=is_infer, hyper=hyper)
                else:
                    xyz, color, opacity, scaling, rot, neural_opacity, mask, normals, semantics, shs_features = generate_neural.generate_neural_gaussians_obj(
                        pipe, viewpoint_camera, model, single_mask, is_training=is_training, stage=stage)
                all_neural_opacity.append(neural_opacity)
                all_mask.append(mask)
                neural_opacity_ids.append(
                    torch.full((neural_opacity.shape[0],), model_idx, dtype=torch.int32, device="cuda"))
            else:
                if model_name == 'background':
                    xyz, color, opacity, scaling, rot, normals, semantics, shs_features = generate_neural.generate_neural_gaussians_kernel(
                        pipe, viewpoint_camera, model, single_mask, is_training=is_training, ape_code=ape_code,
                        stage=stage)
                elif model.obj_class == 'pedestrian':
                    xyz, color, opacity, scaling, rot, normals, semantics, shs_features = generate_neural.generate_neural_gaussians_non_rigid(
                        pipe, viewpoint_camera, model, single_mask, is_training=is_training, ape_code=ape_code,
                        stage=stage, total_frames=total_frames, iteration=iteration, deform=deform, is_infer=is_infer, hyper=hyper)
                else:
                    xyz, color, opacity, scaling, rot, normals, semantics, shs_features = generate_neural.generate_neural_gaussians_obj(
                        pipe, viewpoint_camera, model, single_mask, is_training=is_training, ape_code=ape_code,
                        stage=stage)

            all_xyz.append(xyz)
            all_color.append(color)
            all_opacity.append(opacity)
            all_scaling.append(scaling)
            all_rot.append(rot)
            all_normals.append(normals)
            all_semantics.append(semantics)
            all_ids.append(torch.full((xyz.shape[0],), model_idx, dtype=torch.int32, device="cuda"))
        start_time = time.time()
        def concat_all_features(feature_list, feature_name):
            if not feature_list:
                print(f"Warning: {feature_name} list is empty.")
                return None
            try:
                concatenated = torch.cat(feature_list, dim=0)
                if any(tensor.requires_grad for tensor in feature_list):
                    concatenated.requires_grad_(True)
                return concatenated
            except Exception as e:
                print(f"Error concatenating {feature_name}: {e}")
                return None

        total_xyz = concat_all_features(all_xyz, "all_xyz")
        total_color = concat_all_features(all_color, "all_color")
        total_opacity = concat_all_features(all_opacity, "all_opacity")
        total_scaling = concat_all_features(all_scaling, "all_scaling")
        total_rot = concat_all_features(all_rot, "all_rot")
        total_normals = concat_all_features(all_normals, "all_normals")
        total_semantics = concat_all_features(all_semantics, "all_semantics")
        total_shs_features = None
        total_ids = concat_all_features(all_ids, "all_ids")

        if is_training:
            total_neural_opacity = concat_all_features(all_neural_opacity, "all_neural_opacity")
            total_mask = concat_all_features(all_mask, "all_mask")
        start_time = time.time()

        # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
        screenspace_points = torch.zeros_like(total_xyz, dtype=pc._anchor.dtype, requires_grad=True, device="cuda") + 0

        if retain_grad:
            try:
                screenspace_points.retain_grad()
            except:
                pass

        # Set up rasterization configuration
        tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
        tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

        raster_settings = GaussianRasterizationSettings(
            image_height=int(viewpoint_camera.image_height),
            image_width=int(viewpoint_camera.image_width),
            tanfovx=tanfovx,
            tanfovy=tanfovy,
            bg=bg_color,
            scale_modifier=scaling_modifier,
            viewmatrix=viewpoint_camera.world_view_transform,
            projmatrix=viewpoint_camera.full_proj_transform,
            sh_degree=1,   # 1, pc.active_sh_degree
            campos=viewpoint_camera.camera_center,
            prefiltered=False,
            debug=pipe.debug
        )

        rasterizer = GaussianRasterizer(raster_settings=raster_settings)

        feature_names = []
        feature_dims = []
        features = []

        if pipe.render_normal:
            feature_names.append('normals')
            feature_dims.append(total_normals.shape[-1])
            features.append(total_normals)

        if hyper.use_semantic:
            feature_names.append('semantic')
            feature_dims.append(total_semantics.shape[-1])
            features.append(total_semantics)

        if len(features) > 0:
            features = torch.cat(features, dim=-1)
        else:
            features = None
        rendered_image, radii, depth, rendered_acc, rendered_feature = rasterizer(
            means3D = total_xyz,   # total_xyz
            means2D = screenspace_points,
            shs = None,
            colors_precomp = total_color, # total_color,
            opacities = total_opacity, # total_opacity
            scales = total_scaling, # total_scaling
            rotations = total_rot, # total_rot
            cov3D_precomp = None,
            semantics = features)
        rendered_feature_dict = dict()
        if rendered_feature.shape[0] > 0:
            rendered_feature_list = torch.split(rendered_feature, feature_dims, dim=0)
            for i, feature_name in enumerate(feature_names):
                rendered_feature_dict[feature_name] = rendered_feature_list[i]
        if 'normals' in rendered_feature_dict:
            rendered_feature_dict['normals'] = torch.nn.functional.normalize(rendered_feature_dict['normals'], dim=0)
        if 'semantic' in rendered_feature_dict:
            rendered_semantic = rendered_feature_dict['semantic']
            semantic_mode = hyper.semantic_mode
            if semantic_mode == 'logits':
                pass
            else:
                rendered_semantic = rendered_semantic / (torch.sum(rendered_semantic, dim=0, keepdim=True) + 1e-8) # normalize to probabilities
                rendered_semantic = torch.log(rendered_semantic + 1e-8) # change for cross entropy loss

        # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
        if is_training:
            result = {"render": rendered_image,
                      "acc": rendered_acc,
                    "rendered_depth": depth,
                    "viewspace_points": screenspace_points,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
                    "selection_mask": total_mask,
                    "neural_opacity": total_neural_opacity,
                    "scaling": total_scaling,
                    "total_ids": all_ids,
                    "neural_opacity_ids": neural_opacity_ids,
                    "total_xyz": total_xyz,
                    }
        else:
            result = {"render": rendered_image,
                      "acc": rendered_acc,
                    "rendered_depth": depth,
                    "viewspace_points": screenspace_points,
                    "visibility_filter" : radii > 0,
                    "radii": radii,
                    "ids": total_ids,
                    "neural_opacity_ids": neural_opacity_ids,
                    }
        result.update(rendered_feature_dict)
        return result

def build_rotation(r):
    norm = torch.sqrt(
        r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3]
    )

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device="cuda")

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R
