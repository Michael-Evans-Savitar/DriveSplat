ModelParams = dict(
    init_mode = 'colmap',
    selected_frames = [0, 100],
    white_background = False,
    random_background = True,
    split_test = 4,
    split_train = -1,
    cameras = [0],
    x_quantile = 0.99,
    y_quantile = 0.50,
    z_quantile = 0.01,
    gt_z_use = True,
)

ModelHiddenParams = dict(
    include_sky = False,
    include_obj = True,
    include_bkgd = True,
    appearance_dim = 0,
    use_pose_correction = False,
    use_color_correction = False,

)

OptimizationParams = dict(
    start_stat = 500,
    update_from = 1500,
    update_until = 25_000,
    iterations = 30_000,
    coarse_iterations = 3000,
    update_interval = 100,

    lambda_depth =  0.01,
    lambda_normal = 0.01,
    lambda_semantic = 0.000,
    lambda_reg = 0.0,
    lambda_sky = 0.0,

)
