#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use
#
# For inquiries contact george.drettakis@inria.fr
import os
import glob
import sys
from PIL import Image
from tqdm import tqdm
from typing import NamedTuple
from colorama import Fore, init, Style
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
try:
    import laspy
except:
    print("No laspy")
from scene.gaussian_model import BasicPointCloud
import cv2
import imageio
from utils.waymo_utils import get_obj_pose_tracking, load_camera_info
from utils.waymo_utils import generate_dataparser_outputs
from utils.kitti_utils import generate_kitti_dataparser_outputs
from utils.data_utils import get_val_frames
import torch


class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    K: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    depth: np.array
    normal_gt: np.array
    semantic_gt: np.array
    time : float
    metadata: dict = dict()

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str
    metadata: dict = dict()

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius, "center": center}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    # images_folder
    path = os.path.dirname(images_folder)
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()
        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)
        if intr.model=="SIMPLE_PINHOLE" or intr.model == "SIMPLE_RADIAL":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        # add depth for COLMAP dataset
        depth_path = os.path.join(os.path.dirname(images_folder), "depth/", f"{image_name}.png")
        if os.path.exists(depth_path):
            depth = Image.open(depth_path)
        normal_gt_path = os.path.join(os.path.dirname(images_folder), "normal/", f"{image_name}.jpg")
        if os.path.exists(normal_gt_path):
            normal_gt = Image.open(normal_gt_path)

        semantic_gt = None
        semantic_gt_path = os.path.join(os.path.dirname(images_folder), "dynamic_mask/", f"{image_name}.png")
        if os.path.exists(semantic_gt_path):
            semantic_gt = Image.open(semantic_gt_path)
            semantic_gt = semantic_gt

        K = np.eye(3, 3)
        K[0, 0] = intr.params[0]
        K[1, 1] = intr.params[0]
        K[0, 2] = intr.params[1]
        K[1, 2] = intr.params[2]

        cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, K=K, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height, depth = depth, normal_gt = normal_gt, semantic_gt=semantic_gt, time = float(idx/len(cam_extrinsics)))
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    try:
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    try:
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    except:
        normals = np.random.rand(positions.shape[0], positions.shape[1])
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def fetchLas(path):
    las = laspy.read(path)
    positions = np.vstack((las.x, las.y, las.z)).transpose()
    try:
        colors = np.vstack((las.red, las.green, las.blue)).transpose()
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    normals = np.random.rand(positions.shape[0], positions.shape[1])

    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def read_las_file(path):
    las = laspy.read(path)
    positions = np.vstack((las.x, las.y, las.z)).transpose()
    try:
        colors = np.vstack((las.red, las.green, las.blue)).transpose()
    except:
        colors = np.random.rand(positions.shape[0], positions.shape[1])
    normals = np.random.rand(positions.shape[0], positions.shape[1])

    return positions, colors, normals

def read_multiple_las_files(paths, ply_path):
    all_positions = []
    all_colors = []
    all_normals = []

    for path in paths:
        positions, colors, normals = read_las_file(path)
        all_positions.append(positions)
        all_colors.append(colors)
        all_normals.append(normals)

    all_positions = np.vstack(all_positions)
    all_colors = np.vstack(all_colors)
    all_normals = np.vstack(all_normals)

    print("Saving point cloud to .ply file...")
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    elements = np.empty(all_positions.shape[0], dtype=dtype)
    attributes = np.concatenate((all_positions, all_normals, all_colors), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(ply_path)

    return BasicPointCloud(points=all_positions, colors=all_colors, normals=all_normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    if rgb.max() <= 1. and rgb.min() >= 0:
        rgb = np.clip(rgb * 255, 0., 255.)

    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
             ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
             ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def readColmapSceneInfo(args, hyper, path, images, eval, ds, is_training = True, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
    reading_dir = "images" if ds == 1 else f"images_{ds}"
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    load_dynamic_mask = True
    normal_dir = os.path.join(path, 'normal')
    load_normal = args.use_normal and is_training and os.path.exists(normal_dir)
    depth_dir = os.path.join(path, 'depth')
    load_depth = args.use_depth and is_training and os.path.exists(depth_dir)


    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except:
        cd = None

    # sky mask
    sky_mask_dir = os.path.join(path, 'sky_mask')
    if not os.path.exists(sky_mask_dir):
        cmd = f'python script/waymo/generate_sky_mask.py --datadir {path}'
        print('Generating sky mask')
        os.system(cmd)
    load_sky_mask = is_training

    scene_metadata = dict()
    camera_timestamps = dict()
    for cam in args.cameras:
        camera_timestamps[cam] = dict()
        camera_timestamps[cam]['train_timestamps'] = []
        camera_timestamps[cam]['test_timestamps'] = []

    start_id = args.selected_frames[0]
    end_id = args.selected_frames[1]
    cam_infos_unsorted = []

    for cam in args.cameras:
        camera_timestamps[cam]['train_timestamps'] = sorted(camera_timestamps[cam]['train_timestamps'])
        camera_timestamps[cam]['test_timestamps'] = sorted(camera_timestamps[cam]['test_timestamps'])
    scene_metadata['camera_timestamps'] = camera_timestamps

    if args.mode == 'novel_view':
        nerf_normalization = getNerfppNorm(test_cam_infos)
    else:
        nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_metadata['scene_center'] = nerf_normalization['center']
    scene_metadata['scene_radius'] = nerf_normalization['radius']
    print(f'Scene extent: {nerf_normalization["radius"]}')

    pcd: BasicPointCloud = fetchPly(ply_path)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path,
                           metadata=scene_metadata)
    return scene_info

def readWaymoSceneInfo(args, hyper, path, images, eval, ds, is_training = True, llffhold=8):
    selected_frames = args.selected_frames

    load_dynamic_mask = True
    normal_dir = os.path.join(path, 'normal')
    load_normal = args.use_normal and is_training and os.path.exists(normal_dir)
    depth_dir = os.path.join(path, 'depth')
    load_depth = args.use_depth and is_training and os.path.exists(depth_dir)

    if args.init_mode == 'colmap':
        bkgd_ply_path = os.path.join(args.model_path, "input_ply/points3D_bkgd.ply")
    elif args.init_mode == 'dust3r':
        bkgd_ply_path = os.path.join(args.model_path, "input_ply/points3D_colmap.ply")
    build_pointcloud = is_training and (not os.path.exists(bkgd_ply_path))

    # sky mask
    sky_mask_dir = os.path.join(path, 'sky_mask')
    if not os.path.exists(sky_mask_dir):
        cmd = f'python script/waymo/generate_sky_mask.py --datadir {path}'
        print('Generating sky mask')
        os.system(cmd)
    load_sky_mask = is_training

    # lidar depth
    lidar_depth_dir = os.path.join(path, 'lidar_depth')
    if not os.path.exists(lidar_depth_dir):
        cmd = f'python script/waymo/generate_lidar_depth.py --datadir {path}'
        print('Generating lidar depth')
        os.system(cmd)
    load_lidar_depth = True

    # selected_frames0
    cameras = args.cameras
    output = generate_dataparser_outputs(
        args, hyper,
        datadir=path,
        selected_frames=selected_frames,
        build_pointcloud=build_pointcloud,
        cameras=cameras,
    )
    exts = output['exts']
    ixts = output['ixts']
    poses = output['poses']
    c2ws = output['c2ws']
    image_filenames = output['image_filenames']
    obj_tracklets = output['obj_tracklets']
    obj_info = output['obj_info']
    frames, cams = output['frames'], output['cams']
    frames_idx = output['frames_idx']
    num_frames = output['num_frames']
    cams_timestamps = output['cams_timestamps']
    tracklet_timestamps = output['tracklet_timestamps']
    obj_bounds = output['obj_bounds']

    axis_transform = output['axis_transform']

    train_frames, test_frames = get_val_frames(
        num_frames,
        test_every = args.split_test if args.split_test > 0 else None,
        train_every = args.split_train if args.split_train > 0 else None,
    )

    scene_metadata = dict()
    scene_metadata['obj_tracklets'] = obj_tracklets
    scene_metadata['tracklet_timestamps'] = tracklet_timestamps
    scene_metadata['obj_meta'] = obj_info
    scene_metadata['num_images'] = len(exts)
    scene_metadata['num_cams'] = 1
    scene_metadata['num_frames'] = num_frames
    scene_metadata['axis_transform'] = axis_transform
    camera_timestamps = dict()
    for cam in args.cameras:
        camera_timestamps[cam] = dict()
        camera_timestamps[cam]['train_timestamps'] = []
        camera_timestamps[cam]['test_timestamps'] = []

    cam_infos = []
    for i in tqdm(range(len(exts))):
        # generate pose and image
        ext = exts[i]
        ixt = ixts[i]
        c2w = c2ws[i]
        pose = poses[i]
        image_path = image_filenames[i]
        M_trans = axis_transform[i]
        image_name = os.path.basename(image_path).split('.')[0]
        image = Image.open(image_path)

        width, height = image.size
        fx, fy = ixt[0, 0], ixt[1, 1]
        FovY = focal2fov(fx, height)
        FovX = focal2fov(fy, width)

        RT = np.linalg.inv(c2w)
        R = RT[:3, :3].T
        T = RT[:3, 3]
        K = ixt.copy()

        metadata = dict()
        metadata['frame'] = frames[i]
        metadata['cam'] = cams[i]
        metadata['frame_idx'] = frames_idx[i]
        metadata['ego_pose'] = pose
        metadata['extrinsic'] = ext
        metadata['timestamp'] = cams_timestamps[i]
        metadata['axis_transform'] = M_trans

        if frames_idx[i] in train_frames:
            metadata['is_val'] = False
            camera_timestamps[cams[i]]['train_timestamps'].append(cams_timestamps[i])
        else:
            metadata['is_val'] = True
            camera_timestamps[cams[i]]['test_timestamps'].append(cams_timestamps[i])

        # load dynamic mask
        if load_dynamic_mask:
            metadata['obj_bound'] = Image.fromarray(obj_bounds[i])

        image_name_new = image_name.split('_')[0]

        # Optional: load monocular normal
        if load_normal:
            mono_normal = Image.open(os.path.join(normal_dir, f'{image_name}.jpg'))
            metadata['mono_normal'] = mono_normal

        # Optional load midas depth
        if args.use_depth:
            depth_path = os.path.join(depth_dir, f'{image_name}.npy')
            mono_depth = np.load(depth_path, allow_pickle=True)
            metadata['mono_depth'] = mono_depth
        else:
            mono_depth = None

        if load_sky_mask:
            sky_mask_path = os.path.join(sky_mask_dir, f'{image_name}.png')
            sky_mask = (cv2.imread(sky_mask_path)[..., 0]) > 0.
            sky_mask = Image.fromarray(sky_mask)
            metadata['sky_mask'] = sky_mask

        # load lidar depth
        if load_lidar_depth:
            depth_path = os.path.join(path, 'lidar_depth', f'{image_name}.npy')
            depth_gt = np.load(depth_path, allow_pickle=True)
            if isinstance(depth_gt, np.ndarray):
                depth_gt = dict(depth_gt.item())
                depth_mask = depth_gt['mask']
                print("the number of valid depth: ", np.sum(depth_mask))
                value = depth_gt['value']
                depth_gt = np.zeros_like(depth_mask).astype(np.float32)
                depth_gt[depth_mask] = value

            metadata['lidar_depth'] = depth_gt
        mask = None
        semantic_gt_path = os.path.join(path, 'dynamic_mask', f'{image_name}.png')
        if os.path.exists(semantic_gt_path):
            mask = Image.open(semantic_gt_path)
            mask = mask
        cam_info = CameraInfo(
            uid=i, R=R, T=T, FovY=FovY, FovX=FovX, K=K,
            image=image, image_path=image_path, image_name=image_name,
            width=width, height=height,
            depth=mono_depth, normal_gt=mono_normal, semantic_gt=mask, time=cams_timestamps[i],
            metadata=metadata)
        cam_infos.append(cam_info)

    train_cam_infos = [cam_info for cam_info in cam_infos if not cam_info.metadata['is_val']]
    test_cam_infos = [cam_info for cam_info in cam_infos if cam_info.metadata['is_val']]

    for cam in args.cameras:
        camera_timestamps[cam]['train_timestamps'] = sorted(camera_timestamps[cam]['train_timestamps'])
        camera_timestamps[cam]['test_timestamps'] = sorted(camera_timestamps[cam]['test_timestamps'])
    scene_metadata['camera_timestamps'] = camera_timestamps

    novel_view_cam_infos = []

    if args.mode == 'novel_view':
        nerf_normalization = getNerfppNorm(test_cam_infos)
    else:
        nerf_normalization = getNerfppNorm(train_cam_infos)
    nerf_normalization['radius'] = max(nerf_normalization['radius'], 10)
    scene_metadata['scene_center'] = nerf_normalization['center']
    scene_metadata['scene_radius'] = nerf_normalization['radius']
    print(f'Scene extent: {nerf_normalization["radius"]}')

    pcd: BasicPointCloud = fetchPly(bkgd_ply_path)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=bkgd_ply_path,
                           metadata=scene_metadata)
    return scene_info


def readKITTISceneInfo(args, hyper, path, images, eval, ds, is_training=True, llffhold=8):
    selected_frames = args.selected_frames

    load_dynamic_mask = True
    normal_dir = os.path.join(path, 'normal')
    load_normal = args.use_normal and is_training and os.path.exists(normal_dir)
    depth_dir = os.path.join(path, 'depth')
    load_depth = args.use_depth and is_training and os.path.exists(depth_dir)

    bkgd_ply_path = os.path.join(args.model_path, "input_ply/points3D_bkgd.ply")
    build_pointcloud = is_training and (not os.path.exists(bkgd_ply_path))

    # sky mask
    sky_mask_dir = os.path.join(path, 'sky_mask')
    if not os.path.exists(sky_mask_dir):
        cmd = f'python script/waymo/generate_sky_mask.py --datadir {path}'
        print('Generating sky mask')
        os.system(cmd)
    load_sky_mask = is_training

    # selected_frames0
    cameras = args.cameras
    output = generate_kitti_dataparser_outputs(
        args, hyper,
        datadir=path,
        selected_frames=selected_frames,
        build_pointcloud=build_pointcloud,
        cameras=cameras,
    )
    exts = output['exts']
    ixts = output['ixts']
    poses = output['poses']
    c2ws = output['c2ws']
    image_filenames = output['image_filenames']
    obj_tracklets = output['obj_tracklets']
    obj_info = output['obj_info']
    frames, cams = output['frames'], output['cams']
    frames_idx = output['frames_idx']
    num_frames = output['num_frames']
    cams_timestamps = output['cams_timestamps']
    tracklet_timestamps = output['tracklet_timestamps']
    obj_bounds = output['obj_bounds']
    axis_transform = output['axis_transform']

    train_frames, test_frames = get_val_frames(
        num_frames,
        test_every=args.split_test if args.split_test > 0 else None,
        train_every=args.split_train if args.split_train > 0 else None,
    )

    scene_metadata = dict()
    scene_metadata['obj_tracklets'] = obj_tracklets
    scene_metadata['tracklet_timestamps'] = tracklet_timestamps
    scene_metadata['obj_meta'] = obj_info
    scene_metadata['num_images'] = len(exts)
    scene_metadata['num_cams'] = 1
    scene_metadata['num_frames'] = num_frames
    scene_metadata['axis_transform'] = axis_transform

    camera_timestamps = dict()
    for cam in args.cameras:
        camera_timestamps[cam] = dict()
        camera_timestamps[cam]['train_timestamps'] = []
        camera_timestamps[cam]['test_timestamps'] = []

    cam_infos = []
    for i in tqdm(range(len(exts))):
        # generate pose and image
        ext = exts[i]
        ixt = ixts[i]
        c2w = c2ws[i]
        pose = poses[i]
        image_path = image_filenames[i]
        M_trans = axis_transform[i]
        image_name = os.path.basename(image_path).split('.')[0]
        image = Image.open(image_path)

        width, height = image.size
        fx, fy = ixt[0, 0], ixt[1, 1]
        FovY = focal2fov(fx, height)
        FovX = focal2fov(fy, width)

        RT = np.linalg.inv(c2w)
        R = RT[:3, :3].T
        T = RT[:3, 3]
        K = ixt.copy()

        metadata = dict()
        metadata['frame'] = frames[i]
        metadata['cam'] = cams[i]
        metadata['frame_idx'] = frames_idx[i]
        metadata['ego_pose'] = pose
        metadata['extrinsic'] = ext
        metadata['timestamp'] = cams_timestamps[i]
        metadata['axis_transform'] = M_trans

        if frames_idx[i] in train_frames:
            metadata['is_val'] = False
            camera_timestamps[cams[i]]['train_timestamps'].append(cams_timestamps[i])
        else:
            metadata['is_val'] = True
            camera_timestamps[cams[i]]['test_timestamps'].append(cams_timestamps[i])

        # load dynamic mask
        if load_dynamic_mask:
            metadata['obj_bound'] = Image.fromarray(obj_bounds[i])

        image_name_new = image_name.split('_')[0]

        # Optional: load monocular normal
        if load_normal:
            mono_normal = Image.open(os.path.join(normal_dir, f'{image_name}.jpg'))
            metadata['mono_normal'] = mono_normal

        if args.use_depth:
            depth_path = os.path.join(depth_dir, f'{image_name}.npy')
            mono_depth = np.load(depth_path, allow_pickle=True)
            metadata['mono_depth'] = mono_depth
        else:
            mono_depth = None

        if load_sky_mask:
            sky_mask_path = os.path.join(sky_mask_dir, f'{image_name}.jpg')
            sky_mask = (cv2.imread(sky_mask_path)[..., 0]) > 0.
            sky_mask = Image.fromarray(sky_mask)
            metadata['sky_mask'] = sky_mask

        # load lidar depth
        load_lidar_depth = True
        if load_lidar_depth:
            depth_path = os.path.join(path, 'lidar_depth', f'{image_name_new}.npy')
            depth_gt = np.load(depth_path, allow_pickle=True)
            if isinstance(depth_gt, np.ndarray):
                depth_gt = dict(depth_gt.item())
                depth_mask = depth_gt['mask']
                print("the number of valid depth: ", np.sum(depth_mask))
                value = depth_gt['value']
                depth_gt = np.zeros_like(depth_mask).astype(np.float32)
                depth_gt[depth_mask] = value

            metadata['lidar_depth'] = depth_gt

        mask = None

        cam_info = CameraInfo(
            uid=i, R=R, T=T, FovY=FovY, FovX=FovX, K=K,
            image=image, image_path=image_path, image_name=image_name_new,
            width=width, height=height,
            depth=mono_depth, normal_gt=mono_normal, semantic_gt=mask, time=cams_timestamps[i],
            metadata=metadata)
        cam_infos.append(cam_info)
    train_cam_infos = [cam_info for cam_info in cam_infos if not cam_info.metadata['is_val']]
    test_cam_infos = [cam_info for cam_info in cam_infos if cam_info.metadata['is_val']]

    for cam in args.cameras:
        camera_timestamps[cam]['train_timestamps'] = sorted(camera_timestamps[cam]['train_timestamps'])
        camera_timestamps[cam]['test_timestamps'] = sorted(camera_timestamps[cam]['test_timestamps'])
    scene_metadata['camera_timestamps'] = camera_timestamps

    novel_view_cam_infos = []

    if args.mode == 'novel_view':
        nerf_normalization = getNerfppNorm(test_cam_infos)
    else:
        nerf_normalization = getNerfppNorm(train_cam_infos)

    nerf_normalization['radius'] = max(nerf_normalization['radius'], 10)

    scene_metadata['scene_center'] = nerf_normalization['center']
    scene_metadata['scene_radius'] = nerf_normalization['radius']
    print(f'Scene extent: {nerf_normalization["radius"]}')

    pcd: BasicPointCloud = fetchPly(bkgd_ply_path)
    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=bkgd_ply_path,
                           metadata=scene_metadata
                           )
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Waymo": readWaymoSceneInfo,
    "KITTI": readKITTISceneInfo,
}

def load_depth(tiff_path):
    return imageio.imread(tiff_path)

def load_normal(tiff_path):
    return imageio.imread(tiff_path)

def load_semantic(tiff_path):
    return imageio.imread(tiff_path)

def get_Sphere_Norm(xyz, scale):
    xyz_max = np.max(xyz, axis=0)
    xyz_min = np.min(xyz, axis=0)
    center = (xyz_max + xyz_min) / 2
    radius = np.linalg.norm(xyz_max - xyz_min) / 2.
    radius *= scale

    return {
        'radius': radius,
        'center': center,
    }

def init_depth_anything():
    """DepthAnything"""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = DepthAnything.from_pretrained('LiheYoung/depth_anything_vitl14')
    model.to(device)
    model.eval()
    return model, device

def predict_depth(model, device, image):
    """DepthAnything,metric depth"""
    with torch.no_grad():

        image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        image_tensor = image_tensor.to(device)


        depth = model(image_tensor)
        depth = depth.cpu().numpy().squeeze()

        # ()uint16
        depth_mm = (depth * 1000).astype(np.uint16)
        return Image.fromarray(depth_mm)
