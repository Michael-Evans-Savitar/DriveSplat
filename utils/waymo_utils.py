import os
import numpy as np
import cv2
import torch
import json
import open3d as o3d
import math
from glob import glob
from tqdm import tqdm
from utils.box_utils import bbox_to_corner3d, inbbox_points, get_bound_2d_mask
from utils.colmap_utils import read_points3D_binary, read_extrinsics_binary, qvec2rotmat
from utils.data_utils import get_val_frames
from utils.graphics_utils import get_rays, sphere_intersection
from utils.general_utils import matrix_to_quaternion, quaternion_to_matrix_numpy
from plyfile import PlyData, PlyElement

waymo_track2label = {"vehicle": 0, "pedestrian": 1, "cyclist": 2, "sign": 3, "misc": -1}

_camera2label = {
    'FRONT': 0,
    'FRONT_LEFT': 1,
    'FRONT_RIGHT': 2,
    'SIDE_LEFT': 3,
    'SIDE_RIGHT': 4,
}

_label2camera = {
    0: 'FRONT',
    1: 'FRONT_LEFT',
    2: 'FRONT_RIGHT',
    3: 'SIDE_LEFT',
    4: 'SIDE_RIGHT',
}
image_heights = [1280, 1280, 1280, 886, 886]
image_widths = [1920, 1920, 1920, 1920, 1920]
image_filename_to_cam = lambda x: int(x.split('.')[0][-1])
image_filename_to_frame = lambda x: int(x.split('.')[0][:6])


def _prefer_canonical_pose_path(current_path, candidate_path):
    if current_path is None:
        return candidate_path
    current_frame = os.path.splitext(os.path.basename(current_path))[0].split('_')[0]
    candidate_frame = os.path.splitext(os.path.basename(candidate_path))[0].split('_')[0]
    current_is_canonical = len(current_frame) == 6
    candidate_is_canonical = len(candidate_frame) == 6
    if candidate_is_canonical and not current_is_canonical:
        return candidate_path
    return current_path

# load ego pose and camera calibration(extrinsic and intrinsic)
def load_camera_info(datadir):
    ego_pose_dir = os.path.join(datadir, 'ego_pose')
    extrinsics_dir = os.path.join(datadir, 'extrinsics')
    intrinsics_dir = os.path.join(datadir, 'intrinsics')

    intrinsics = []
    extrinsics = []
    for i in range(5):
        intrinsic = np.loadtxt(os.path.join(intrinsics_dir,  f"{i}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics.append(intrinsic)

    for i in range(5):
        cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir,  f"{i}.txt"))
        extrinsics.append(cam_to_ego)

    ego_frame_pose_paths = {}
    ego_cam_pose_paths = [dict() for i in range(5)]
    ignored_duplicate_pose_files = 0
    ego_pose_paths = sorted(os.listdir(ego_pose_dir))
    for ego_pose_path in ego_pose_paths:
        stem = os.path.splitext(ego_pose_path)[0]
        tokens = stem.split('_')
        try:
            frame_id = int(tokens[0])
        except ValueError:
            continue
        # frame pose (, 000192.txt). Some processed scenes contain both
        # 000.txt and 000000.txt; keep one canonical path per numeric frame id.
        if len(tokens) == 1:
            previous = ego_frame_pose_paths.get(frame_id)
            preferred = _prefer_canonical_pose_path(previous, ego_pose_path)
            if previous is not None:
                ignored_duplicate_pose_files += 1
            ego_frame_pose_paths[frame_id] = preferred
        # camera pose (, 000192_4.txt)
        else:
            try:
                cam = int(tokens[-1])
            except ValueError:
                continue
            if cam < 0 or cam >= 5:
                continue
            previous = ego_cam_pose_paths[cam].get(frame_id)
            preferred = _prefer_canonical_pose_path(previous, ego_pose_path)
            if previous is not None:
                ignored_duplicate_pose_files += 1
            ego_cam_pose_paths[cam][frame_id] = preferred

    if ignored_duplicate_pose_files > 0:
        print(f"Ignored duplicate ego pose files after numeric frame-id canonicalization: {ignored_duplicate_pose_files}")

    frame_ids = sorted(ego_frame_pose_paths.keys())
    ego_frame_poses = []
    for frame_id in frame_ids:
        ego_frame_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_frame_pose_paths[frame_id]))
        ego_frame_poses.append(ego_frame_pose)

    ego_cam_poses = [[] for i in range(5)]
    for cam_id in range(5):
        for frame_id in frame_ids:
            if frame_id in ego_cam_pose_paths[cam_id]:
                ego_cam_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_cam_pose_paths[cam_id][frame_id]))
                ego_cam_poses[cam_id].append(ego_cam_pose)

    # pose,0posepose
    if len(ego_frame_poses) == 0 and len(ego_cam_poses[0]) > 0:
        print("No separate frame pose files found, deriving from camera 0 poses...")
        frame_ids = sorted(ego_cam_pose_paths[0].keys())
        cam0_extrinsic = extrinsics[0]  # cam_to_ego
        cam0_extrinsic_inv = np.linalg.inv(cam0_extrinsic)
        for frame_id in frame_ids:
            ego_cam_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_cam_pose_paths[0][frame_id]))
            ego_frame_pose = ego_cam_pose @ cam0_extrinsic_inv
            ego_frame_poses.append(ego_frame_pose)
        ego_cam_poses = [[] for i in range(5)]
        for cam_id in range(5):
            for frame_id in frame_ids:
                if frame_id in ego_cam_pose_paths[cam_id]:
                    ego_cam_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_cam_pose_paths[cam_id][frame_id]))
                    ego_cam_poses[cam_id].append(ego_cam_pose)

    # center ego pose
    ego_frame_poses = np.array(ego_frame_poses)
    print(f"the shape of ego_frame_poses: {ego_frame_poses.shape}")
    center_point = np.mean(ego_frame_poses[:, :3, 3], axis=0)
    ego_frame_poses[:, :3, 3] -= center_point # [num_frames, 4, 4]

    # pose
    num_frames = ego_frame_poses.shape[0]
    cam_pose_counts = [len(ego_cam_poses[i]) for i in range(5)]
    has_complete_cam_poses = all(count == num_frames for count in cam_pose_counts)

    if has_complete_cam_poses:
        print(f"Using camera pose files directly. Camera pose counts: {cam_pose_counts}")
        ego_cam_poses = [np.array(ego_cam_poses[i]) for i in range(5)]
        ego_cam_poses = np.stack(ego_cam_poses, axis=0)  # [5, num_frames, 4, 4]
        ego_cam_poses[:, :, :3, 3] -= center_point
    else:
        # pose,poseextrinsics
        print(f"Camera pose files incomplete or inconsistent (counts: {cam_pose_counts}, expected: {num_frames})")
        print("Computing camera poses from frame poses and extrinsics...")
        ego_cam_poses = np.zeros((5, num_frames, 4, 4))
        for cam_id in range(5):
            cam_to_ego = extrinsics[cam_id]
            for frame_id in range(num_frames):
                ego_cam_poses[cam_id, frame_id] = ego_frame_poses[frame_id] @ cam_to_ego

    return intrinsics, extrinsics, ego_frame_poses, ego_cam_poses

# calculate obj pose in world frame
# box_info: box_center_x box_center_y box_center_z box_heading
def make_obj_pose(ego_pose, box_info):
    tx, ty, tz, heading = box_info
    c = math.cos(heading)
    s = math.sin(heading)
    rotz_matrix = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    obj_pose_vehicle = np.eye(4)
    obj_pose_vehicle[:3, :3] = rotz_matrix
    obj_pose_vehicle[:3, 3] = np.array([tx, ty, tz])
    obj_pose_world = np.matmul(ego_pose, obj_pose_vehicle)

    obj_rotation_vehicle = torch.from_numpy(obj_pose_vehicle[:3, :3]).float().unsqueeze(0)
    obj_quaternion_vehicle = matrix_to_quaternion(obj_rotation_vehicle).squeeze(0).numpy()
    obj_quaternion_vehicle = obj_quaternion_vehicle / np.linalg.norm(obj_quaternion_vehicle)
    obj_position_vehicle = obj_pose_vehicle[:3, 3]
    obj_pose_vehicle = np.concatenate([obj_position_vehicle, obj_quaternion_vehicle])

    obj_rotation_world = torch.from_numpy(obj_pose_world[:3, :3]).float().unsqueeze(0)
    obj_quaternion_world = matrix_to_quaternion(obj_rotation_world).squeeze(0).numpy()
    obj_quaternion_world = obj_quaternion_world / np.linalg.norm(obj_quaternion_world)
    obj_position_world = obj_pose_world[:3, 3]
    obj_pose_world = np.concatenate([obj_position_world, obj_quaternion_world])

    return obj_pose_vehicle, obj_pose_world




def get_obj_pose_tracking(args, hyper, datadir, selected_frames, ego_poses, cameras=[0, 1, 2, 3, 4]):
    tracklets_ls = []
    objects_info = {}

    if args.use_tracker:
        tracklet_path = os.path.join(datadir, 'track/track_info_castrack.txt')
        tracklet_camera_vis_path = os.path.join(datadir, 'track/track_camera_vis_castrack.json')
    else:
        tracklet_path = os.path.join(datadir, 'track/track_info.txt')
        tracklet_camera_vis_path = os.path.join(datadir, 'track/track_camera_vis.json')

    print(f'Loading from : {tracklet_path}')
    f = open(tracklet_path, 'r')
    tracklets_str = f.read().splitlines()
    tracklets_str = tracklets_str[1:]

    # track_camera_vis "";


    f = open(tracklet_camera_vis_path, 'r')
    tracklet_camera_vis = json.load(f)

    start_frame, end_frame = selected_frames[0], selected_frames[1]

    image_dir = os.path.join(datadir, 'images')
    n_cameras = len(cameras)
    n_images = len(os.listdir(image_dir))
    n_frames = n_images // n_cameras
    n_obj_in_frame = np.zeros(n_frames)

    for tracklet in tracklets_str:
        tracklet = tracklet.split()
        frame_id = int(tracklet[0])
        track_id = int(tracklet[1])
        object_class = tracklet[2]

        if object_class in ['sign', 'misc']:
            continue

        cameras_vis_list = []
        try:
            cameras_vis_list = tracklet_camera_vis[str(track_id)][str(frame_id)]
        except Exception:
            cameras_vis_list = []



        if len(cameras_vis_list) > 0:

            join_cameras_list = list(set(cameras) & set(cameras_vis_list))
            if len(join_cameras_list) == 0:
                continue
        else:


            continue

        if track_id not in objects_info.keys():
            objects_info[track_id] = dict()
            objects_info[track_id]['track_id'] = track_id
            objects_info[track_id]['class'] = object_class
            objects_info[track_id]['class_label'] = waymo_track2label[object_class]
            objects_info[track_id]['height'] = float(tracklet[4])
            objects_info[track_id]['width'] = float(tracklet[5])
            objects_info[track_id]['length'] = float(tracklet[6])
        else:
            objects_info[track_id]['height'] = max(objects_info[track_id]['height'], float(tracklet[4]))
            objects_info[track_id]['width'] = max(objects_info[track_id]['width'], float(tracklet[5]))
            objects_info[track_id]['length'] = max(objects_info[track_id]['length'], float(tracklet[6]))

        # track_info.txt :
        # frame_id track_id object_class alpha h w l cx cy cz heading (speed)
        #  alpha  3 ,:tracklet[6:10] -> (cx,cy,cz,heading)
        tr_array = np.concatenate(
            [
                np.array(tracklet[:2]).astype(np.float64),
                np.array([float(tracklet[3])], dtype=np.float64),
                np.array(tracklet[4:]).astype(np.float64),
            ]
        )
        tracklets_ls.append(tr_array)
        n_obj_in_frame[frame_id] += 1

    tracklets_array = np.array(tracklets_ls)
    max_obj_per_frame = int(n_obj_in_frame[start_frame:end_frame + 1].max())
    num_frames = end_frame - start_frame + 1
    visible_objects_ids = np.ones([num_frames, max_obj_per_frame]) * -1.0
    visible_objects_pose_vehicle = np.ones([num_frames, max_obj_per_frame, 7]) * -1.0
    visible_objects_pose_world = np.ones([num_frames, max_obj_per_frame, 7]) * -1.0

    # Iterate through the tracklets and process object data
    for tracklet in tracklets_array:
        frame_id = int(tracklet[0])
        track_id = int(tracklet[1])
        if start_frame <= frame_id <= end_frame:
            ego_pose = ego_poses[frame_id]
            obj_pose_vehicle, obj_pose_world = make_obj_pose(ego_pose, tracklet[6:10])

            frame_idx = frame_id - start_frame
            obj_column = np.argwhere(visible_objects_ids[frame_idx, :] < 0).min()

            visible_objects_ids[frame_idx, obj_column] = track_id
            visible_objects_pose_vehicle[frame_idx, obj_column] = obj_pose_vehicle
            visible_objects_pose_world[frame_idx, obj_column] = obj_pose_world

    # Remove static objects
    print("Removing static objects")
    for key in objects_info.copy().keys():
        all_obj_idx = np.where(visible_objects_ids == key)
        if len(all_obj_idx[0]) > 0:
            obj_world_postions = visible_objects_pose_world[all_obj_idx][:, :3]
            distance = np.linalg.norm(obj_world_postions[0] - obj_world_postions[-1])
            dynamic = np.any(np.std(obj_world_postions, axis=0) > 0.5) or distance > 2
            if not dynamic:
                visible_objects_ids[all_obj_idx] = -1.
                visible_objects_pose_vehicle[all_obj_idx] = -1.
                visible_objects_pose_world[all_obj_idx] = -1.
                objects_info.pop(key)
        else:
            objects_info.pop(key)

    # Clip max_num_obj
    mask = visible_objects_ids >= 0
    max_obj_per_frame_new = np.sum(mask, axis=1).max()
    print("Max obj per frame:", max_obj_per_frame_new)

    if max_obj_per_frame_new == 0:
        print("No moving objects in current sequence; using empty visible-object placeholders")
        visible_objects_ids = np.ones([num_frames, 1]) * -1.0
        visible_objects_pose_world = np.ones([num_frames, 1, 7]) * -1.0
        visible_objects_pose_vehicle = np.ones([num_frames, 1, 7]) * -1.0
    elif max_obj_per_frame_new < max_obj_per_frame:
        visible_objects_ids_new = np.ones([num_frames, max_obj_per_frame_new]) * -1.0
        visible_objects_pose_vehicle_new = np.ones([num_frames, max_obj_per_frame_new, 7]) * -1.0
        visible_objects_pose_world_new = np.ones([num_frames, max_obj_per_frame_new, 7]) * -1.0
        for frame_idx in range(num_frames):
            for y in range(max_obj_per_frame):
                obj_id = visible_objects_ids[frame_idx, y]
                if obj_id >= 0:
                    obj_column = np.argwhere(visible_objects_ids_new[frame_idx, :] < 0).min()
                    visible_objects_ids_new[frame_idx, obj_column] = obj_id
                    visible_objects_pose_vehicle_new[frame_idx, obj_column] = visible_objects_pose_vehicle[frame_idx, y]
                    visible_objects_pose_world_new[frame_idx, obj_column] = visible_objects_pose_world[frame_idx, y]

        visible_objects_ids = visible_objects_ids_new
        visible_objects_pose_vehicle = visible_objects_pose_vehicle_new
        visible_objects_pose_world = visible_objects_pose_world_new

    box_scale = hyper.box_scale
    print('box scale: ', box_scale)

    frames = list(range(start_frame, end_frame + 1))
    frames = np.array(frames).astype(np.int32)

    # postprocess object_info
    for key in objects_info.keys():
        obj = objects_info[key]
        if obj['class'] == 'Cyclist' or obj['class'] == 'pedestrian':
            obj['deformable'] = True
        else:
            obj['deformable'] = False

        obj['width'] = obj['width'] * box_scale
        obj['length'] = obj['length'] * box_scale

        obj_frame_idx = np.argwhere(visible_objects_ids == key)[:, 0]
        obj_frame_idx = obj_frame_idx.astype(np.int32)
        obj_frames = frames[obj_frame_idx]
        obj['start_frame'] = np.min(obj_frames)
        obj['end_frame'] = np.max(obj_frames)

        objects_info[key] = obj

    # [num_frames, max_obj, track_id, x, y, z, qw, qx, qy, qz]
    objects_tracklets_world = np.concatenate(
        [visible_objects_ids[..., None], visible_objects_pose_world], axis=-1
    )

    objects_tracklets_vehicle = np.concatenate(
        [visible_objects_ids[..., None], visible_objects_pose_vehicle], axis=-1
    )


    return objects_tracklets_world, objects_tracklets_vehicle, objects_info



def padding_tracklets(tracklets, frame_timestamps, min_timestamp, max_timestamp):
    # tracklets: [num_frames, max_obj, ....]
    # frame_timestamps: [num_frames]

    # Clone instead of extrapolation
    if min_timestamp < frame_timestamps[0]:
        tracklets_first = tracklets[0]
        frame_timestamps = np.concatenate([[min_timestamp], frame_timestamps])
        tracklets = np.concatenate([tracklets_first[None], tracklets], axis=0)

    if max_timestamp > frame_timestamps[-1]:
        tracklets_last = tracklets[-1]
        frame_timestamps = np.concatenate([frame_timestamps, [max_timestamp]])
        tracklets = np.concatenate([tracklets, tracklets_last[None]], axis=0)

    return tracklets, frame_timestamps

def detect_road_line(points, num_lines, line_percent, is_source=False):
    """"""

    z_values = points[:, 2]
    z_hist, z_bins = np.histogram(z_values, bins=100)
    z_ground = z_bins[np.argmax(z_hist)]
    z_bin_width = z_bins[1] - z_bins[0]

    ground_mask = np.abs(points[:, 2] - z_ground) < z_bin_width * 2
    ground_points = points[ground_mask]

    if len(ground_points) < 100:
        return None

    ground_x_coords = ground_points[:, 0]
    ground_y_coords = ground_points[:, 1]
    ground_width = np.percentile(ground_x_coords, 95) - np.percentile(ground_x_coords, 5)
    ground_x_center = np.median(ground_x_coords)


    y_sorted = np.sort(ground_y_coords)
    total_points = len(ground_points)
    lines_info = []

    for i in range(num_lines):
        start_idx = int(total_points * i * line_percent)
        end_idx = int(total_points * (i + 1) * line_percent)

        if start_idx >= len(y_sorted) or end_idx >= len(y_sorted):
            return None

        y_threshold_start = y_sorted[start_idx]
        y_threshold_end = y_sorted[end_idx]

        line_mask = (ground_points[:, 1] >= y_threshold_start) & (ground_points[:, 1] < y_threshold_end)
        line_points = ground_points[line_mask]

        if len(line_points) < 30:
            return None

        x_coords = line_points[:, 0]
        x_mean = np.mean(x_coords)
        x_std = np.std(x_coords)
        x_inlier_mask = np.abs(x_coords - x_mean) < 2 * x_std
        filtered_points = line_points[x_inlier_mask]

        if len(filtered_points) < 20:
            return None

        x_min = np.percentile(filtered_points[:, 0], 10)
        x_max = np.percentile(filtered_points[:, 0], 90)
        x_center = (x_max + x_min) / 2
        length = x_max - x_min
        y_pos = np.mean(filtered_points[:, 1])

        lines_info.append((x_center, y_pos, length))

    return lines_info, ground_width, ground_x_center, z_ground

def score_alignment(source, target, transform):
    """"""
    scale, rotation, translation = transform
    source_aligned = scale * source + translation


    z_values = source_aligned[:, 2]
    z_hist, z_bins = np.histogram(z_values, bins=100)
    source_ground_z = z_bins[np.argmax(z_hist)]

    z_values = target[:, 2]
    z_hist, z_bins = np.histogram(z_values, bins=100)
    target_ground_z = z_bins[np.argmax(z_hist)]


    source_ground_mask = np.abs(source_aligned[:, 2] - source_ground_z) < 0.1
    target_ground_mask = np.abs(target[:, 2] - target_ground_z) < 0.1

    source_ground = source_aligned[source_ground_mask]
    target_ground = target[target_ground_mask]


    scores = {}


    scores['ground_z'] = abs(source_ground_z - target_ground_z)


    source_front_y = np.percentile(source_ground[:, 1], 1)
    target_front_y = np.percentile(target_ground[:, 1], 1)
    scores['front_y'] = abs(source_front_y - target_front_y)


    source_center_x = np.median(source_ground[:, 0])
    target_center_x = np.median(target_ground[:, 0])
    scores['center_x'] = abs(source_center_x - target_center_x)


    source_width = np.percentile(source_ground[:, 0], 95) - np.percentile(source_ground[:, 0], 5)
    target_width = np.percentile(target_ground[:, 0], 95) - np.percentile(target_ground[:, 0], 5)
    scores['width_ratio'] = abs(1 - source_width/target_width)


    total_score = (scores['ground_z'] * 10 +
                  scores['front_y'] * 10 +
                  scores['center_x'] * 5 +
                  scores['width_ratio'] * 5)

    return total_score, scores

def verify_point_cloud_quality(points, is_source=False):
    ""","""
    try:

        num_points = len(points)
        if num_points < 1000:
            return False, f"Too few points: {num_points}"


        x_range = np.max(points[:, 0]) - np.min(points[:, 0])
        y_range = np.max(points[:, 1]) - np.min(points[:, 1])
        z_range = np.max(points[:, 2]) - np.min(points[:, 2])

        print(f"\nPoint cloud stats ({'source' if is_source else 'target'}):")
        print(f"Number of points: {num_points}")
        print(f"X range: {x_range:.2f}")
        print(f"Y range: {y_range:.2f}")
        print(f"Z range: {z_range:.2f}")

        if x_range < 1.0 or y_range < 1.0 or z_range < 0.5:
            return False, f"Point cloud range too small: x={x_range:.2f}, y={y_range:.2f}, z={z_range:.2f}"

        if x_range > 100.0 or y_range > 100.0 or z_range > 50.0:
            return False, f"Point cloud range too large: x={x_range:.2f}, y={y_range:.2f}, z={z_range:.2f}"


        area = x_range * y_range
        density = num_points / area
        print(f"Point density: {density:.2f} points/m")


        min_density = 2.0 if is_source else 1.0
        if density < min_density:
            return False, f"Point cloud density too low: {density:.2f} points/m (minimum required: {min_density})"


        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return False, "Invalid values (NaN or Inf) in point cloud"


        std_x = np.std(points[:, 0])
        std_y = np.std(points[:, 1])
        std_z = np.std(points[:, 2])

        print(f"Standard deviations: x={std_x:.2f}, y={std_y:.2f}, z={std_z:.2f}")


        if std_x < 0.05 or std_y < 0.05 or std_z < 0.05:
            return False, f"Point distribution too concentrated: std_x={std_x:.2f}, std_y={std_y:.2f}, std_z={std_z:.2f}"

        return True, "Point cloud quality check passed"

    except Exception as e:
        return False, f"Error in point cloud verification: {str(e)}"

def verify_and_adjust_front_alignment(source_aligned, target, transform):
    """"""
    try:

        source_front_y = np.min(source_aligned[:, 1])
        target_front_y = np.min(target[:, 1])


        source_front_mask = source_aligned[:, 1] <= source_front_y + 0.1
        target_front_mask = target[:, 1] <= target_front_y + 0.1

        if not np.any(source_front_mask) or not np.any(target_front_mask):
            print("No front points found, skipping adjustment")
            return source_aligned, transform

        source_front_x = np.mean(source_aligned[source_front_mask][:, 0])
        target_front_x = np.mean(target[target_front_mask][:, 0])


        y_diff = target_front_y - source_front_y
        x_diff = target_front_x - source_front_x

        print(f"\nFront alignment check:")
        print(f"Y difference: {y_diff:.3f}")
        print(f"X difference: {x_diff:.3f}")


        if abs(y_diff) > 10.0 or abs(x_diff) > 5.0:
            print("Front difference too large, skipping adjustment")
            return source_aligned, transform


        y_threshold = 0.5
        x_threshold = 0.3

        if abs(y_diff) > y_threshold or abs(x_diff) > x_threshold:
            print("Adjusting front alignment...")


            y_adjust = np.clip(y_diff * 0.2, -1.0, 1.0)
            x_adjust = np.clip(x_diff * 0.2, -0.5, 0.5)


            scale, rotation, translation = transform
            new_translation = translation.copy()
            new_translation[1] += y_adjust
            new_translation[0] += x_adjust


            new_transform = (scale, rotation, new_translation)
            new_source_aligned = scale * source_aligned + new_translation - translation


            new_score, new_details = score_alignment(new_source_aligned, target, new_transform)
            old_score, old_details = score_alignment(source_aligned, target, transform)

            print(f"Original score: {old_score:.3f}")
            print(f"New score: {new_score:.3f}")
            print(f"Adjustment: X={x_adjust:.3f}, Y={y_adjust:.3f}")


            if new_score < old_score * 2.0:
                print("Front alignment adjustment accepted")
                return new_source_aligned, new_transform
            else:
                print("Front alignment adjustment rejected (score too high)")
                return source_aligned, transform
        else:
            print("Front alignment is good, no adjustment needed")
            return source_aligned, transform

    except Exception as e:
        print(f"Error in front alignment verification: {str(e)}")
        return source_aligned, transform

def coarse_align_pointclouds(source, target, camera_poses=None):
    """"""
    try:
        if camera_poses is not None:
            source_transformed = transform_points_to_world(source.copy(), camera_poses, None)
        else:
            source_transformed = source.copy()


        source_results = detect_road_line(source_transformed, num_lines=7,
                                       line_percent=0.015, is_source=True)
        target_results = detect_road_line(target, num_lines=7,
                                       line_percent=0.015, is_source=False)

        if source_results is None or target_results is None:
            return None, None

        source_lines, source_width, source_x_center, source_z = source_results
        target_lines, target_width, target_x_center, target_z = target_results


        scales = {
            'width': target_width / source_width,
            'lines': [target_lines[i][2] / source_lines[i][2] for i in range(7)],
            'spacing': []
        }


        for i in range(6):
            source_spacing = source_lines[i+1][1] - source_lines[i][1]
            target_spacing = target_lines[i+1][1] - target_lines[i][1]
            if abs(source_spacing) > 1e-6:
                scales['spacing'].append(target_spacing / source_spacing)


        scale_weights = {
            'width': 0.4,
            'lines': 0.3,
            'spacing': 0.3
        }


        scale = (scales['width'] * scale_weights['width'] +
                np.mean(scales['lines']) * scale_weights['lines'] +
                np.mean(scales['spacing']) * scale_weights['spacing'])

        if scale < 8.0 or scale > 35.0:
            return None, None


        translation = np.zeros(3)


        x_trans_ground = target_x_center - (scale * source_x_center)
        x_trans_lines = [target_lines[i][0] - (scale * source_lines[i][0]) for i in range(7)]
        translation[0] = x_trans_ground * 0.7 + sum(t * w for t, w in zip(x_trans_lines, [0.3/6] * 7))


        y_trans = [target_lines[i][1] - (scale * source_lines[i][1]) for i in range(7)]
        translation[1] = sum(t * w for t, w in zip(y_trans, [0.4, 0.3, 0.15, 0.1, 0.05, 0, 0]))


        translation[2] = target_z - (scale * source_z)


        transform = (scale, np.eye(3), translation)
        score, details = score_alignment(source_transformed, target, transform)


        if score < float('inf'):

            width_error = abs(1 - scales['width']/scale)
            lines_error = abs(1 - np.mean(scales['lines'])/scale)
            spacing_error = abs(1 - np.mean(scales['spacing'])/scale)

            total_error = width_error + lines_error + spacing_error
            if total_error > 0:

                scale_weights['width'] = (1 - width_error/total_error) / 3
                scale_weights['lines'] = (1 - lines_error/total_error) / 3
                scale_weights['spacing'] = (1 - spacing_error/total_error) / 3


        source_aligned = scale * source_transformed + translation


        source_aligned, transform = verify_and_adjust_front_alignment(source_aligned, target, transform)

        return source_aligned, transform

    except Exception as e:
        print(f"Error in point cloud alignment: {str(e)}")
        return None, None

def coarse_align_backup(source, target):
    """Fallback point-cloud alignment used when COLMAP poses are insufficient."""
    try:
        print("[Waymo] Using centroid/percentile fallback alignment")

        source_centroid = np.mean(source, axis=0)
        target_centroid = np.mean(target, axis=0)
        source_scale = np.sqrt(np.mean(np.sum((source - source_centroid)**2, axis=1)))
        target_scale = np.sqrt(np.mean(np.sum((target - target_centroid)**2, axis=1)))
        scale = target_scale / source_scale


        translation = np.zeros(3)


        percentile = 99
        source_y_front = np.percentile(source[:, 1], percentile)
        target_y_front = np.percentile(target[:, 1], percentile)
        source_front_mask = source[:, 1] >= source_y_front - 0.2
        target_front_mask = target[:, 1] >= target_y_front - 0.2
        source_front_mean = np.mean(source[source_front_mask, 1])
        target_front_mean = np.mean(target[target_front_mask, 1])
        translation[1] = target_front_mean - (scale * source_front_mean)


        translation[0] = target_centroid[0] - (scale * source_centroid[0])


        percentile = 5
        source_z_bottom = np.percentile(source[:, 2], percentile)
        target_z_bottom = np.percentile(target[:, 2], percentile)
        source_bottom_mask = source[:, 2] <= source_z_bottom
        target_bottom_mask = target[:, 2] <= target_z_bottom
        source_bottom_mean = np.mean(source[source_bottom_mask, 2])
        target_bottom_mean = np.mean(target[target_bottom_mask, 2])
        translation[2] = target_bottom_mean - (scale * source_bottom_mean)

        print(f"[Waymo] Fallback alignment: scale={scale:.3f}, translation={translation}")


        source_aligned = scale * source + translation
        rotation = np.eye(3)

        return source_aligned, (scale, rotation, translation)
    except Exception as e:
        print(f"Error in backup align method: {str(e)}")

        return source, (1.0, np.eye(3), np.zeros(3))

def generate_dataparser_outputs(
        args, hyper,
        datadir,
        selected_frames=None,
        build_pointcloud=True,
        cameras=[0, 1, 2, 3, 4]
):
    image_dir = os.path.join(datadir, 'images')
    image_filenames_all = sorted(glob(os.path.join(image_dir, '*.png')))
    num_frames_all = len(image_filenames_all) // 5
    num_cameras = len(cameras)

    if selected_frames is None:
        start_frame = 0
        end_frame = num_frames_all - 1
        selected_frames = [start_frame, end_frame]
    else:
        start_frame, end_frame = selected_frames[0], selected_frames[1]
    num_frames = end_frame - start_frame + 1

    # load calibration and ego pose
    intrinsics, extrinsics, ego_frame_poses, ego_cam_poses = load_camera_info(datadir)

    # load camera, frame, path
    frames = []
    frames_idx = []
    cams = []
    image_filenames = []

    ixts = []
    exts = []
    poses = []
    c2ws = []

    frames_timestamps = []
    cams_timestamps = []

    split_test = args.split_test
    split_train = args.split_train
    train_frames, test_frames = get_val_frames(
        num_frames,
        test_every=split_test if split_test > 0 else None,
        train_every=split_train if split_train > 0 else None,
    )

    timestamp_path = os.path.join(datadir, 'timestamps.json')
    with open(timestamp_path, 'r') as f:
        timestamps = json.load(f)

    for frame in range(start_frame, end_frame + 1):
        frames_timestamps.append(timestamps['FRAME'][f'{frame:06d}'])

    for image_filename in image_filenames_all:
        image_basename = os.path.basename(image_filename)
        frame = image_filename_to_frame(image_basename)
        cam = image_filename_to_cam(image_basename)
        if frame >= start_frame and frame <= end_frame and cam in cameras:
            ixt = intrinsics[cam]
            ext = extrinsics[cam]
            pose = ego_cam_poses[cam, frame]
            c2w = pose @ ext

            frames.append(frame)
            frames_idx.append(frame - start_frame)
            cams.append(cam)
            image_filenames.append(image_filename)

            ixts.append(ixt)
            exts.append(ext)
            poses.append(pose)
            c2ws.append(c2w)

            camera_name = _label2camera[cam]
            timestamp = timestamps[camera_name][f'{frame:06d}']
            cams_timestamps.append(timestamp)

    exts = np.stack(exts, axis=0)
    ixts = np.stack(ixts, axis=0)
    poses = np.stack(poses, axis=0)
    c2ws = np.stack(c2ws, axis=0)

    timestamp_offset = min(cams_timestamps + frames_timestamps)
    cams_timestamps = np.array(cams_timestamps) - timestamp_offset
    frames_timestamps = np.array(frames_timestamps) - timestamp_offset
    min_timestamp, max_timestamp = min(cams_timestamps.min(), frames_timestamps.min()), max(cams_timestamps.max(),
                                                                                            frames_timestamps.max())

    _, object_tracklets_vehicle, object_info = get_obj_pose_tracking(
        args, hyper,
        datadir,
        selected_frames,
        ego_frame_poses,
        cameras,
    )

    for track_id in object_info.keys():
        object_start_frame = object_info[track_id]['start_frame']
        object_end_frame = object_info[track_id]['end_frame']
        object_start_timestamp = timestamps['FRAME'][f'{object_start_frame:06d}'] - timestamp_offset - 0.1
        object_end_timestamp = timestamps['FRAME'][f'{object_end_frame:06d}'] - timestamp_offset + 0.1
        object_info[track_id]['start_timestamp'] = max(object_start_timestamp, min_timestamp)
        object_info[track_id]['end_timestamp'] = min(object_end_timestamp, max_timestamp)

    result = dict()
    result['num_frames'] = num_frames
    result['exts'] = exts
    result['ixts'] = ixts
    result['poses'] = poses
    result['c2ws'] = c2ws
    result['obj_tracklets'] = object_tracklets_vehicle
    result['obj_info'] = object_info
    result['frames'] = frames
    result['cams'] = cams
    result['frames_idx'] = frames_idx
    result['image_filenames'] = image_filenames
    result['cams_timestamps'] = cams_timestamps
    result['tracklet_timestamps'] = frames_timestamps

    # get object bounding mask
    obj_bounds = []
    m_trans = []
    for i, image_filename in tqdm(enumerate(image_filenames)):
        cam = cams[i]
        h, w = image_heights[cam], image_widths[cam]
        obj_bound = np.zeros((h, w)).astype(np.uint8)
        obj_tracklets = object_tracklets_vehicle[frames_idx[i]]
        ixt, ext = ixts[i], exts[i]
        mm = np.eye(4, 4)
        m_trans.append(mm)
        for obj_tracklet in obj_tracklets:
            track_id = int(obj_tracklet[0])
            if track_id >= 0:
                obj_pose_vehicle = np.eye(4)
                obj_pose_vehicle[:3, :3] = quaternion_to_matrix_numpy(obj_tracklet[4:8])
                obj_pose_vehicle[:3, 3] = obj_tracklet[1:4]
                obj_length = object_info[track_id]['length']
                obj_width = object_info[track_id]['width']
                obj_height = object_info[track_id]['height']
                bbox = np.array([[-obj_length, -obj_width, -obj_height],
                                 [obj_length, obj_width, obj_height]]) * 0.5
                corners_local = bbox_to_corner3d(bbox)
                corners_local = np.concatenate([corners_local, np.ones_like(corners_local[..., :1])], axis=-1)
                corners_vehicle = corners_local @ obj_pose_vehicle.T  # 3D bounding box in vehicle frame
                mask = get_bound_2d_mask(
                    corners_3d=corners_vehicle[..., :3],
                    K=ixt,
                    pose=np.linalg.inv(ext),
                    H=h, W=w
                )
                obj_bound = np.logical_or(obj_bound, mask)
        obj_bounds.append(obj_bound)
    result['obj_bounds'] = obj_bounds

    # run colmap
    if args.init_mode == 'colmap':
        colmap_basedir = os.path.join(f'{args.model_path}', 'colmap')
        if not os.path.exists(os.path.join(colmap_basedir, 'triangulated/sparse/model')):
            from utils.colmap_waymo_full import run_colmap_waymo
            run_colmap_waymo(result, args)
    else:
        colmap_basedir = os.path.join(f'{args.model_path}', 'dust3r')
        if not os.path.exists(os.path.join(colmap_basedir, 'triangulated/sparse/model/points3D.bin')):
            from dust3r_waymo_full import run_dust3r_waymo
            run_dust3r_waymo(result, args)

    result['axis_transform'] = m_trans
    if build_pointcloud:
        print('build point cloud')
        pointcloud_dir = os.path.join(f'{args.model_path}', 'input_ply')
        os.makedirs(pointcloud_dir, exist_ok=True)

        points_xyz_dict = dict()
        points_rgb_dict = dict()
        points_xyz_dict['bkgd'] = []
        points_rgb_dict['bkgd'] = []
        for track_id in object_info.keys():
            points_xyz_dict[f'obj_{track_id:03d}'] = []
            points_rgb_dict[f'obj_{track_id:03d}'] = []

        print('initialize from sfm pointcloud')
        points_colmap_path = os.path.join(colmap_basedir, 'triangulated/sparse/model/points3D.bin')
        points_colmap_xyz, points_colmap_rgb, points_colmap_error = read_points3D_binary(points_colmap_path)
        points_colmap_rgb = points_colmap_rgb / 255.

        print('initialize from lidar pointcloud')
        pointcloud_path = os.path.join(datadir, 'pointcloud.npz')
        if os.path.exists(pointcloud_path):
            # ====== ,copy ======
            pts3d_dict = np.load(pointcloud_path, allow_pickle=True)['pointcloud'].item()
            pts2d_dict = np.load(pointcloud_path, allow_pickle=True)['camera_projection'].item()

            for i, frame in tqdm(enumerate(range(start_frame, end_frame + 1))):
                idxs = list(range(i * num_cameras, (i + 1) * num_cameras))
                cams_frame = [cams[idx] for idx in idxs]
                image_filenames_frame = [image_filenames[idx] for idx in idxs]

                raw_3d = pts3d_dict[frame]
                raw_2d = pts2d_dict[frame]

                # use the first projection camera
                points_camera_all = raw_2d[..., 0]
                points_projw_all = raw_2d[..., 1]
                points_projh_all = raw_2d[..., 2]

                mask = np.array([c in cameras for c in points_camera_all]).astype(np.bool_)

                points_xyz_vehicle = raw_3d[mask]
                ego_pose = ego_frame_poses[frame]
                points_xyz_vehicle = np.concatenate(
                    [points_xyz_vehicle,
                     np.ones_like(points_xyz_vehicle[..., :1])], axis=-1
                )
                points_xyz_world = points_xyz_vehicle @ ego_pose.T

                points_rgb = np.ones_like(points_xyz_vehicle[:, :3])
                points_camera = points_camera_all[mask]
                points_projw = points_projw_all[mask]
                points_projh = points_projh_all[mask]

                for cam, image_filename in zip(cams_frame, image_filenames_frame):
                    mask_cam = (points_camera == cam)
                    image = cv2.imread(image_filename)[..., [2, 1, 0]] / 255.

                    mask_projw = points_projw[mask_cam]
                    mask_projh = points_projh[mask_cam]
                    mask_rgb = image[mask_projh, mask_projw]
                    points_rgb[mask_cam] = mask_rgb

                points_xyz_obj_mask = np.zeros(points_xyz_vehicle.shape[0], dtype=np.bool_)

                for tracklet in object_tracklets_vehicle[i]:
                    track_id = int(tracklet[0])
                    if track_id >= 0:
                        obj_pose_vehicle = np.eye(4)
                        obj_pose_vehicle[:3, :3] = quaternion_to_matrix_numpy(tracklet[4:8])
                        obj_pose_vehicle[:3, 3] = tracklet[1:4]
                        vehicle2local = np.linalg.inv(obj_pose_vehicle)

                        points_xyz_obj = points_xyz_vehicle @ vehicle2local.T
                        points_xyz_obj = points_xyz_obj[..., :3]

                        length = object_info[track_id]['length']
                        width = object_info[track_id]['width']
                        height = object_info[track_id]['height']
                        bbox = [[-length / 2, -width / 2, -height / 2], [length / 2, width / 2, height / 2]]
                        obj_corners_3d_local = bbox_to_corner3d(bbox)

                        points_xyz_inbbox = inbbox_points(points_xyz_obj, obj_corners_3d_local)
                        points_xyz_obj_mask = np.logical_or(points_xyz_obj_mask, points_xyz_inbbox)
                        points_xyz_dict[f'obj_{track_id:03d}'].append(points_xyz_obj[points_xyz_inbbox])
                        points_rgb_dict[f'obj_{track_id:03d}'].append(points_rgb[points_xyz_inbbox])

                points_lidar_xyz = points_xyz_world[~points_xyz_obj_mask][..., :3]
                points_lidar_rgb = points_rgb[~points_xyz_obj_mask]

                points_xyz_dict['bkgd'].append(points_lidar_xyz)
                points_rgb_dict['bkgd'].append(points_lidar_rgb)

            initial_num_obj = 5000

            for k, v in points_xyz_dict.items():
                if len(v) == 0:
                    continue
                else:
                    points_xyz = np.concatenate(v, axis=0)
                    points_rgb = np.concatenate(points_rgb_dict[k], axis=0)
                    if k == 'bkgd':
                        points_lidar = o3d.geometry.PointCloud()
                        points_lidar.points = o3d.utility.Vector3dVector(points_xyz)
                        points_lidar.colors = o3d.utility.Vector3dVector(points_rgb)
                        downsample_points_lidar = points_lidar.voxel_down_sample(voxel_size=0.15)
                        downsample_points_lidar, _ = downsample_points_lidar.remove_radius_outlier(nb_points=10, radius=0.5)
                        points_lidar_xyz = np.asarray(downsample_points_lidar.points).astype(np.float32)
                        points_lidar_rgb = np.asarray(downsample_points_lidar.colors).astype(np.float32)
                    elif k.startswith('obj'):
                        if len(points_xyz) > initial_num_obj:
                            random_indices = np.random.choice(len(points_xyz), initial_num_obj, replace=False)
                            points_xyz = points_xyz[random_indices]
                            points_rgb = points_rgb[random_indices]
                        points_xyz_dict[k] = points_xyz
                        points_rgb_dict[k] = points_rgb
                    else:
                        raise NotImplementedError()
            lidar_sphere_normalization = get_Sphere_Norm(points_lidar_xyz)
            sphere_center = lidar_sphere_normalization['center']
            sphere_radius = lidar_sphere_normalization['radius']

            try:
                if hyper.filter_colmap:
                    points_colmap_mask = np.ones(points_colmap_xyz.shape[0], dtype=np.bool_)
                    for i, ext in enumerate(exts):
                        camera_position = c2ws[i][:3, 3]
                        radius = np.linalg.norm(points_colmap_xyz - camera_position, axis=-1)
                        mask = np.logical_or(radius < hyper.extent, points_colmap_xyz[:, 2] < camera_position[2])
                        points_colmap_mask = np.logical_and(points_colmap_mask, ~mask)
                    points_colmap_xyz = points_colmap_xyz[points_colmap_mask]
                    points_colmap_rgb = points_colmap_rgb[points_colmap_mask]

                points_colmap_dist = np.linalg.norm(points_colmap_xyz - sphere_center, axis=-1)
                mask = points_colmap_dist < 2 * sphere_radius
                points_colmap_xyz = points_colmap_xyz[mask]
                points_colmap_rgb = points_colmap_rgb[mask]

                points_bkgd_xyz = np.concatenate([points_lidar_xyz, points_colmap_xyz], axis=0)
                points_bkgd_rgb = np.concatenate([points_lidar_rgb, points_colmap_rgb], axis=0)
            except:
                print('No colmap pointcloud')
                points_bkgd_xyz = points_lidar_xyz
                points_bkgd_rgb = points_lidar_rgb

            points_xyz_dict['lidar'] = points_lidar_xyz
            points_rgb_dict['lidar'] = points_lidar_rgb
            points_xyz_dict['colmap'] = points_colmap_xyz
            points_rgb_dict['colmap'] = points_colmap_rgb
            points_xyz_dict['bkgd'] = points_bkgd_xyz
            points_rgb_dict['bkgd'] = points_bkgd_rgb

            result['points_xyz_dict'] = points_xyz_dict
            result['points_rgb_dict'] = points_rgb_dict

            for k in points_xyz_dict.keys():
                points_xyz = points_xyz_dict[k]
                points_rgb = points_rgb_dict[k]
                ply_path = os.path.join(pointcloud_dir, f'points3D_{k}.ply')
                try:
                    storePly(ply_path, points_xyz, points_rgb)
                    print(f'saving pointcloud for {k}, number of initial points is {points_xyz.shape}')
                except:
                    print(f'failed to save pointcloud for {k}')
                    continue

        else:
            print('No lidar pointcloud found, initializing dynamic points randomly in bbox')
            initial_num_obj = 200
            for i, frame in tqdm(enumerate(range(start_frame, end_frame + 1))):
                for tracklet in object_tracklets_vehicle[i]:
                    track_id = int(tracklet[0])
                    if track_id >= 0:

                        length = object_info[track_id]['length']
                        width = object_info[track_id]['width']
                        height = object_info[track_id]['height']
                        num_points = initial_num_obj
                        points_local = np.random.uniform(
                            low=[-length/2, -width/2, -height/2],
                            high=[length/2, width/2, height/2],
                            size=(num_points, 3)
                        )
                        # local -> vehicle
                        obj_pose_vehicle = np.eye(4)
                        obj_pose_vehicle[:3, :3] = quaternion_to_matrix_numpy(tracklet[4:8])
                        obj_pose_vehicle[:3, 3] = tracklet[1:4]
                        points_local_homo = np.concatenate([points_local, np.ones((num_points, 1))], axis=1)
                        points_vehicle = points_local_homo @ obj_pose_vehicle.T
                        points_vehicle = points_vehicle[:, :3]
                        # vehicle -> world
                        ego_pose = ego_frame_poses[frame]
                        points_vehicle_homo = np.concatenate([points_vehicle, np.ones((num_points, 1))], axis=1)
                        points_world = points_vehicle_homo @ ego_pose.T
                        points_world = points_world[:, :3]
                        # world -> vehicle
                        ego_pose_inv = np.linalg.inv(ego_pose)
                        points_world_homo = np.concatenate([points_world, np.ones((num_points, 1))], axis=1)
                        points_vehicle2 = points_world_homo @ ego_pose_inv.T
                        points_vehicle2 = points_vehicle2[:, :3]
                        # vehicle -> local
                        obj_pose_vehicle_inv = np.linalg.inv(obj_pose_vehicle)
                        points_vehicle2_homo = np.concatenate([points_vehicle2, np.ones((num_points, 1))], axis=1)
                        points_local_final = points_vehicle2_homo @ obj_pose_vehicle_inv.T
                        points_local_final = points_local_final[:, :3]

                        points_rgb = np.random.uniform(0, 1, size=(num_points, 3))
                        # local
                        points_xyz_dict[f'obj_{track_id:03d}'].append(points_local_final)
                        points_rgb_dict[f'obj_{track_id:03d}'].append(points_rgb)
            # colmap
            points_xyz_dict['bkgd'] = [points_colmap_xyz]
            points_rgb_dict['bkgd'] = [points_colmap_rgb]
            points_xyz_dict['colmap'] = [points_colmap_xyz]
            points_rgb_dict['colmap'] = [points_colmap_rgb]
            #  ply
            for k in points_xyz_dict.keys():
                if len(points_xyz_dict[k]) == 0:
                    continue
                points_xyz = np.concatenate(points_xyz_dict[k], axis=0)
                points_rgb = np.concatenate(points_rgb_dict[k], axis=0)
                ply_path = os.path.join(pointcloud_dir, f'points3D_{k}.ply')
                try:
                    points_rgb = (points_rgb * 255.0).astype(np.uint8)
                    storePly(ply_path, points_xyz, points_rgb)
                    print(f'saving pointcloud for {k}, number of initial points is {points_xyz.shape}')
                except:
                    print(f'failed to save pointcloud for {k}')
                    continue

    return result


def invalid_to_nans(arr, valid_mask, ndim=999):
    if valid_mask is not None:
        arr = arr.clone()
        arr[~valid_mask] = float('nan')
    if arr.ndim > ndim:
        arr = arr.flatten(-2 - (arr.ndim - ndim), -2)
    return arr

def get_joint_pointcloud_depth(pts, valid_masks=None, quantile=0.5, axis=2):
    """

    Args:
        pts: ,:
            - , (N, 3)
            -  (N, 3)
        valid_masks: ,None
        quantile: ,0.5()
        axis: ,0=x, 1=y, 2=z()

    Returns:
        depth:
    """
    # torch tensor
    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts).float()
    elif isinstance(pts, list):
        pts = [torch.from_numpy(p).float() if isinstance(p, np.ndarray) else p for p in pts]


    if isinstance(pts, torch.Tensor) and pts.ndim == 2:
        pts = [pts]
        if valid_masks is not None:
            valid_masks = [valid_masks]

    # NaN
    _coords = []
    for i in range(len(pts)):
        valid_mask = valid_masks[i] if valid_masks is not None else None
        _coord = invalid_to_nans(pts[i], valid_mask)
        if _coord.ndim == 2:
            _coord = _coord.unsqueeze(0)  # batch(1, N, 3)
        _coord = _coord[..., axis]
        _coords.append(_coord)

    # batch
    _coords = torch.cat(_coords, dim=1)


    if quantile == 0.5:
        depth = torch.nanmedian(_coords, dim=1).values
    else:
        depth = torch.nanquantile(_coords, quantile, dim=1)


    depth = depth.view(-1, 1, 1)

    return depth



def get_Sphere_Norm(xyz, scale=1):
    xyz_max = np.max(xyz, axis=0)
    xyz_min = np.min(xyz, axis=0)
    center = (xyz_max + xyz_min) / 2
    radius = np.linalg.norm(xyz_max - xyz_min) / 2.
    radius *= scale

    return {
        'radius': radius,
        'center': center,
    }

def get_joint_pointcloud_center_scale(pts, valid_masks=None, z_only=False, center=True):
    """

    Args:
        pts: ,:
            - , (N, 3)
            -  (N, 3)
        valid_masks: ,None
        z_only: z
        center:

    Returns:
        center:
        scale:
    """
    # torch tensor
    if isinstance(pts, np.ndarray):
        pts = torch.from_numpy(pts).float()
    elif isinstance(pts, list):
        pts = [torch.from_numpy(p).float() if isinstance(p, np.ndarray) else p for p in pts]


    if isinstance(pts, torch.Tensor) and pts.ndim == 2:
        pts = [pts]
        if valid_masks is not None:
            valid_masks = [valid_masks]

    # NaN
    _pts = []
    for i in range(len(pts)):
        valid_mask = valid_masks[i] if valid_masks is not None else None
        _pt = invalid_to_nans(pts[i], valid_mask)
        if _pt.ndim == 2:
            _pt = _pt.unsqueeze(0)  # batch(1, N, 3)
        _pts.append(_pt)

    # batch
    _pts = torch.cat(_pts, dim=1)

    # (nanmedian)
    _center = torch.nanmedian(_pts, dim=1, keepdim=True).values
    if z_only:
        _center[..., :2] = 0


    if center:
        _pts_centered = _pts - _center
    else:
        _pts_centered = _pts
    _norm = _pts_centered.norm(dim=-1)

    # (nanmedian)
    scale = torch.nanmedian(_norm, dim=1).values


    _center = _center.unsqueeze(1)
    scale = scale.view(-1, 1, 1, 1)

    return _center, scale

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
