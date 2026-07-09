#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup):
    def __init__(self, parser, sentinel=False):
        self.feat_dim = 32
        self.n_offsets = 10
        self.fork = 2

        self.mode = "train"  # novel_view

        self.use_feat_bank = False
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self.white_background = True  # False
        self.random_background = False
        self.resolution_scales = [1.0]

        self.data_device = "cuda"
        self.eval = False    # False
        self.ds = 1
        self.ratio = 1 # sampling the input point cloud
        self.undistorted = False

        self.appearance_dim = 0
        self.add_opacity_dist = False
        self.add_cov_dist = False
        self.add_color_dist = False
        self.add_level = False

        self.extend = 1.1
        self.dist2level = 'round'
        self.base_layer = -1 # -1(adaptive) or 10 (default) or 0 ~
        self.visible_threshold = 0.01 # -1(adaptive) or 0.0 ~ 1.0
        self.update_ratio = 0.2

        self.progressive = False
        self.dist_ratio = 0.999
        self.levels = -1
        self.init_level = -1
        self.extra_ratio = 0.25
        self.extra_up = 0.01
        self.sh_degree = 3
        self.selected_frames = [50, 148]
        self.use_tracker = False
        self.split_test = -1
        self.split_train = 1
        self.cameras = [0]
        self.use_normal = True
        self.use_depth = True
        self.use_colmap_pose = True
        self.init_mode = 'colmap'
        self.x_quantile = 0.50
        self.y_quantile = 0.12
        self.z_quantile = 0.01
        self.gt_z_use = False


        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.compute_cov3D_python = False
        self.debug = False
        self.convert_SHs_python = False
        self.render_normal = True
        self.num_classes_global = 1
        super().__init__(parser, "Pipeline Parameters")

class ModelHiddenParams(ParamGroup):
    def __init__(self, parser):
        self.net_width = 64
        self.semantic_mode = 'logits' #'logits'
        self.num_classes = 0
        self.fourier_dim = 1
        self.fourier_scale = 1.
        self.flip_prob = 0.
        self.opt_track = True # True
        self.use_semantic = True # True
        self.include_obj = True # True
        self.include_bkgd = True
        self.include_sky = False
        self.use_pose_correction = False
        self.use_color_correction = False
        self.resolution_sky = 1024
        self.sky_white_background = True
        self.box_scale = 1.5
        self.extent = 10
        self.filter_colmap = False  # True
        self.pose_mode = "image"
        self.color_mode = "image"
        self.color_use_mlp = True

        self.non_rigid_start_iter = 0  # Start using non-rigid deformation after this iteration (0 = from start)
        self.non_rigid_warmup_iter = 2000  # Gradually introduce non-rigid deformation over this many iterations
        self.max_lod_level = 8

        # Anchor-conditioned deformation (AGD) is the default non-rigid actor
        # path. It predicts deformation in anchor space and propagates it to
        # per-Gaussian offsets.
        self.use_anchor_deform = True
        self.anchor_deform_feat_dim = 32
        self.anchor_deform_hidden = 256
        self.anchor_deform_layers = 8
        self.anchor_deform_use_rotation = True
        self.anchor_deform_xyz_multires = 10
        self.anchor_deform_t_multires = 10

        # Hash encoding parameters for neural anchor features.
        self.use_hash_encoding = True

        # General hash parameters
        self.hash_levels = -1
        self.anchor_generation_levels = None
        self.hash_blend_mode = "replace"
        self.hash_blend_weight = 0.5
        self.hash_feat_dim = 32
        self.hash_log2_size = 19
        self.hash_base_resolution = 16
        self.hash_finest_resolution = 2048
        self.hash_disable_lod = False
        self.use_hash_feat_single_level = True
        self.use_tcnn_hash = True

        # Background hash parameters.
        self.hash_base_resolution_bkgd = 256
        self.hash_levels_bkgd = 10
        self.hash_finest_resolution_bkgd = 16384
        self.hash_log2_size_bkgd = 21

        # Object hash parameters.
        self.hash_base_resolution_obj = 16
        self.hash_levels_obj = 5
        self.hash_finest_resolution_obj = 512
        self.hash_log2_size_obj = 17

        # Common hash feature settings.
        self.hash_features_per_level = 2
        self.use_hash_feat_single_level = True
        self.use_hash_feat_multi_level = False
        self.use_hash_interpolation = True

        # Per-component overrides. By default, background and object models use
        # their standard neural Gaussian features while the shared hash feature
        # path remains enabled.
        self.use_hash_feat_single_level_bkgd = False
        self.use_hash_feat_multi_level_bkgd = False
        self.use_hash_feat_single_level_obj = False
        self.use_hash_feat_multi_level_obj = False

        self.hash_disable_lod = False

        # Optional per-anchor LOD bias.
        self.use_lod_bias = False
        self.lod_bias_lr = 0.001
        self.lod_adaptive_strategy = "Ud"
        self.lod_warmup_iters_bias = 1000

        # Hash LOD partitioner. max_hash_level is inferred from the active
        # background/object hash-level settings unless a scene config overrides it.
        self.use_hash_lod_partitioner = True
        def _pos(v):
            try:
                iv = int(v)
                return iv if iv > 0 else 0
            except Exception:
                return 0

        self.max_hash_level = max(_pos(self.hash_levels), _pos(self.hash_levels_bkgd), _pos(self.hash_levels_obj))

        super().__init__(parser, "ModelHiddenParams")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.coarse_iterations = 3000
        self.position_lr_init = 0.00016
        self.position_lr_final = 0.0000016
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = self.iterations
        self.deform_lr_max_steps = 30_000

        self.offset_lr_init = 0.01
        self.offset_lr_final = 0.0001
        self.offset_lr_delay_mult = 0.01
        self.offset_lr_max_steps = self.iterations

        self.feature_lr = 0.0075
        self.opacity_lr = 0.02
        self.scaling_lr = 0.007
        self.rotation_lr = 0.002
        # Flow

        self.mlp_opacity_lr_init = 0.002
        self.mlp_opacity_lr_final = 0.00002
        self.mlp_opacity_lr_delay_mult = 0.01
        self.mlp_opacity_lr_max_steps = self.iterations

        self.mlp_cov_lr_init = 0.004
        self.mlp_cov_lr_final = 0.004
        self.mlp_cov_lr_delay_mult = 0.01
        self.mlp_cov_lr_max_steps = self.iterations

        self.mlp_color_lr_init = 0.008
        self.mlp_color_lr_final = 0.00005
        self.mlp_color_lr_delay_mult = 0.01
        self.mlp_color_lr_max_steps = self.iterations

        self.mlp_featurebank_lr_init = 0.01
        self.mlp_featurebank_lr_final = 0.00001
        self.mlp_featurebank_lr_delay_mult = 0.01
        self.mlp_featurebank_lr_max_steps = self.iterations

        self.appearance_lr_init = 0.01
        self.appearance_lr_final = 0.0001
        self.appearance_lr_delay_mult = 0.01
        self.appearance_lr_max_steps = self.iterations

        self.percent_dense = 0.01
        self.lambda_dssim = 0.2

        self.start_stat = 500
        self.update_from = 1500
        self.coarse_iter = 10000
        self.coarse_factor = 1.5
        self.update_interval = 100
        self.update_until = 25000
        self.update_anchor = True

        self.min_opacity = 0.005
        self.success_threshold = 0.8
        self.densify_grad_threshold = 0.0002
        self.densify_grad_threshold_bkgd = 0.0002
        self.densify_grad_threshold_obj = 0.0002
        self.deformation_lr_init = 0.000016
        self.deformation_lr_final = 0.0000016
        self.deformation_lr_delay_mult = 0.01
        self.grid_lr_init = 0.003
        self.grid_lr_final = 0.0001
        self.semantic_lr = 0.01

        self.lambda_depth =  0.01
        self.lambda_lidar_depth = 0.0
        self.lambda_normal = 0.01
        self.lambda_semantic = 0.00
        self.lambda_reg = 0.0
        self.lambda_sky = 0.0
        self.lambda_sky_scale = [1, 1, 0]
        self.lambda_color_correction = 0.0
        self.lambda_pose_correction = 0.0

        self.track_position_lr_delay_mult = 0.01
        self.track_position_lr_init = 0.005
        self.track_position_lr_final = 5.0e-5
        self.track_position_max_steps = 30000

        self.track_rotation_lr_delay_mult = 0.01
        self.track_rotation_lr_init = 0.001
        self.track_rotation_lr_final = 1.0e-5
        self.track_rotation_max_steps = 30000
        self.opacity_reset_interval = 3000

        self.sky_cube_map_lr_init = 0.01
        self.sky_cube_map_lr_final = 0.0001

        self.pose_correction_lr_init = 5e-6
        self.pose_correction_lr_final = 1e-6
        self.pose_correction_max_steps = 0.01
        self.pose_correction_lr_delay_mult = 0.01


        self.color_correction_lr_init = 5e-4
        self.color_correction_lr_final = 5e-5
        self.color_correction_lr_delay_mult = 0.01
        self.color_correction_max_steps = 0.01

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)
