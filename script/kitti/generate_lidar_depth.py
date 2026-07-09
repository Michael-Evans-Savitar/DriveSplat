import sys
import os
sys.path.append(os.getcwd())
import argparse
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
from utils.img_utils import visualize_depth_numpy

# lambda
frame_id_from_bin = lambda x: int(os.path.basename(x).split('.')[0])
image_filename_to_cam = lambda x: int(x.split('.')[0][-1])
cam_id_from_image = lambda x: int(x.split('.')[0][-1])
frame_id_from_image = lambda x: int(x.split('.')[0][:6])

def load_kitti_calib(calib_dir):
    extrinsics_dir = os.path.join(calib_dir, 'extrinsics')
    intrinsics_dir = os.path.join(calib_dir, 'intrinsics')

    intrinsics = []
    extrinsics = []
    for i in range(2):
        print(f"load path: {os.path.join(intrinsics_dir,  f'{i}.txt')}")
        intrinsic = np.loadtxt(os.path.join(intrinsics_dir,  f"{i}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics.append(intrinsic)
        cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir,  f"{i}.txt"))

    for i in range(2):
        cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir,  f"{i}.txt"))
        extrinsics.append(cam_to_ego)

    return extrinsics, intrinsics

def load_calibration(datadir):
    extrinsics_dir = os.path.join(datadir, 'extrinsics')
    intrinsics_dir = os.path.join(datadir, 'intrinsics')

    intrinsics = []
    extrinsics = []
    for i in range(2):
        intrinsic = np.loadtxt(os.path.join(intrinsics_dir,  f"{i}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics.append(intrinsic)
        cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir,  f"{i}.txt"))

    for i in range(2):
        cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir,  f"{i}.txt"))
        extrinsics.append(cam_to_ego)

    return extrinsics, intrinsics


import os
import numpy as np
import cv2

def bin_to_depth(bin_path, calib, img_shape=(1242, 375)):
    """"""

    datadir = os.path.dirname(os.path.dirname(bin_path))
    save_dir = os.path.join(datadir, 'lidar_depth')
    os.makedirs(save_dir, exist_ok=True)


    frame = frame_id_from_bin(bin_path)
    points = np.fromfile(bin_path, dtype=np.float32).reshape(-1, 4)[:, :3]


    image_file_name = f'{frame:03d}_0.jpg'
    image_path = os.path.join(datadir, 'images', image_file_name)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image file not found: {image_path}")
    h, w = image.shape[:2]


    extrinsics, intrinsics = load_calibration(datadir)
    cam = image_filename_to_cam(image_file_name)


    c2w = extrinsics[cam]
    w2c = np.linalg.inv(c2w)
    points_xyz_cam = points @ w2c[:3, :3].T + w2c[:3, 3]
    points_depth = points_xyz_cam[..., 2]
    valid_mask = points_depth > 0.


    points_xyz_pixel = points_xyz_cam @ intrinsics[cam].T
    points_xyz_pixel[:, :2] /= points_xyz_pixel[:, 2:3]
    points_coord = points_xyz_pixel[valid_mask, :2].round().astype(np.int32)
    points_coord[:, 0] = np.clip(points_coord[:, 0], 0, w - 1)
    points_coord[:, 1] = np.clip(points_coord[:, 1], 0, h - 1)


    depth = np.full((h, w), np.finfo(np.float32).max, dtype=np.float32).reshape(-1)
    u, v = points_coord[:, 0], points_coord[:, 1]
    indices = v * w + u
    np.minimum.at(depth, indices, points_depth[valid_mask])
    depth[depth >= np.finfo(np.float32).max - 1e-5] = 0


    depth_path = os.path.join(save_dir, f'{frame:03d}.npy')
    depth_file = {'mask': (depth != 0).reshape(h, w), 'value': depth[depth != 0]}
    np.save(depth_path, depth_file)


    depth_vis_path = os.path.join(save_dir, f'{frame:03d}.png')
    try:
        if cam == 0:
            depth_vis, _ = visualize_depth_numpy(depth.reshape(h, w))
            depth_on_img = image[..., [2, 1, 0]]
            depth_on_img[depth > 0] = depth_vis[depth > 0]
            cv2.imwrite(depth_vis_path, depth_on_img)
    except:
        print(f'Error in visualizing depth for {image_file_name}, depth range: {depth.min()} - {depth.max()}')

    return depth_file


def generate_lidar_depth_per_frame(datadir):
    """"""
    save_dir = os.path.join(datadir, 'lidar_depth')
    os.makedirs(save_dir, exist_ok=True)


    calib = load_kitti_calib(datadir)

    # bin
    bin_files = sorted(glob(os.path.join(datadir, 'lidar', '*.bin')))


    for bin_path in tqdm(bin_files):
        frame_id = frame_id_from_bin(bin_path)
        depth_map = bin_to_depth(bin_path, calib)

        # npy
        np.save(os.path.join(save_dir, f'{frame_id:03d}.npy'), depth_map)


        # depth_vis, _ = visualize_depth_numpy(depth_map)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', required=True, type=str)
    args = parser.parse_args()

    generate_lidar_depth_per_frame(args.datadir)
