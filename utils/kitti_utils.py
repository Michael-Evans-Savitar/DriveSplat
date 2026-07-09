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

kitti_track2label = {"Car": 0, "Tram": 1, "Cyclist": 2, "pedestrian": 3, "misc": -1}
#     # Rigid objects (vehicles)
#     'Car': ModelType.RigidNodes,
#     'Van': ModelType.RigidNodes,
#     'Truck': ModelType.RigidNodes,
#     'Tram': ModelType.RigidNodes,
#
#     # Humans (SMPL model)
#     'Pedestrian': ModelType.SMPLNodes,
#     'Person_sitting': ModelType.SMPLNodes,
#
#     # Potentially deformable objects
#     'Cyclist': ModelType.DeformableNodes,


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
image_filename_to_frame = lambda x: int(x.split('.')[0][:3])

# load ego pose and camera calibration(extrinsic and intrinsic)
def load_camera_info(datadir):
    ego_pose_dir = os.path.join(datadir, 'ego_pose')
    extrinsics_dir = os.path.join(datadir, 'extrinsics')
    intrinsics_dir = os.path.join(datadir, 'intrinsics')

    intrinsics = []
    extrinsics = []
    for i in range(2):
        intrinsic = np.loadtxt(os.path.join(intrinsics_dir,  f"{i}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics.append(intrinsic)

    for i in range(2):
        cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir,  f"{i}.txt"))
        extrinsics.append(cam_to_ego)

    ego_frame_poses = []
    ego_cam_poses = [[] for i in range(2)]
    ego_pose_paths = sorted(os.listdir(ego_pose_dir))
    for ego_pose_path in ego_pose_paths:

        # frame pose
        if '_' not in ego_pose_path:
            ego_frame_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
            ego_frame_poses.append(ego_frame_pose)
        else:
            cam = image_filename_to_cam(ego_pose_path)
            ego_cam_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
            assert ego_cam_pose.shape == (4, 4), " 4x4 ."
            ego_cam_poses[cam].append(ego_cam_pose)

    # center ego pose
    ego_frame_poses = np.array(ego_frame_poses)
    center_point = np.mean(ego_frame_poses[:, :3, 3], axis=0)
    ego_frame_poses[:, :3, 3] -= center_point # [num_frames, 4, 4]

    ego_cam_poses = [np.array(ego_cam_poses[i]) for i in range(2)]
    #         ego_cam_poses[i].append(np.zeros((4, 4)))
    ego_cam_poses = np.array(ego_cam_poses)
    print("the shape of ego_cam_poses:", ego_cam_poses.shape)
    print("the value of ego_cam_poses:", ego_cam_poses)
    ego_cam_poses[:, :, :3, 3] -= center_point # [5, num_frames, 4, 4]
    return intrinsics, extrinsics, ego_frame_poses, ego_cam_poses


def load_kitti_camera_info(datadir):
    ego_pose_dir = os.path.join(datadir, 'ego_pose')
    extrinsics_dir = os.path.join(datadir, 'extrinsics')
    intrinsics_dir = os.path.join(datadir, 'intrinsics')

    intrinsics = []
    extrinsics = []
    for i in range(1):
        intrinsic = np.loadtxt(os.path.join(intrinsics_dir, f"{i}.txt"))
        fx, fy, cx, cy = intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        intrinsic = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        intrinsics.append(intrinsic)

    for i in range(1):
        cam_to_ego = np.loadtxt(os.path.join(extrinsics_dir, f"{i}.txt"))
        extrinsics.append(cam_to_ego)

    ego_frame_poses = []
    ego_cam_poses = [[] for i in range(1)]
    ego_pose_paths = sorted(os.listdir(ego_pose_dir))
    for ego_pose_path in ego_pose_paths:

        # frame pose
        if '_' not in ego_pose_path:
            ego_frame_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
            ego_frame_poses.append(ego_frame_pose)
        else:
            cam = image_filename_to_cam(ego_pose_path)
            ego_cam_pose = np.loadtxt(os.path.join(ego_pose_dir, ego_pose_path))
            ego_cam_poses[cam].append(ego_cam_pose)

    # center ego pose
    ego_frame_poses = np.array(ego_frame_poses)
    center_point = np.mean(ego_frame_poses[:, :3, 3], axis=0)
    ego_frame_poses[:, :3, 3] -= center_point  # [num_frames, 4, 4]

    ego_cam_poses = [np.array(ego_cam_poses[i]) for i in range(1)]
    ego_cam_poses = np.array(ego_cam_poses)
    ego_cam_poses[:, :, :3, 3] -= center_point  # [5, num_frames, 4, 4]
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




def get_obj_pose_tracking(args, hyper, datadir, selected_frames, ego_poses, cameras=[0, 1]):
    tracklets_ls = []
    objects_info = {}
    start_frame, end_frame = selected_frames[0], selected_frames[1]
    if args.use_tracker:
        tracklet_path = os.path.join(datadir, 'instances/track_info_castrack.txt')
        tracklet_camera_vis_path = os.path.join(datadir, 'instances/track_camera_vis_castrack.json')
    else:
        instances_info_path = os.path.join(datadir, 'instances/instances_info.json')
        frame_instances_path = os.path.join(datadir, 'instances/frame_instances.json')
        tracklet_path = os.path.join(datadir, 'instances/track_info.txt')
        tracklet_camera_vis_path = os.path.join(datadir, 'instances/track_camera_vis.json')

    with open(instances_info_path, "r") as f:
        instances_info = json.load(f)
    with open(frame_instances_path, "r") as f:
        frame_instances = json.load(f)
    num_instances = len(instances_info)
    num_full_frames = len(frame_instances)
    instances_pose = np.zeros((num_full_frames, num_instances, 4, 4))
    instances_size = np.zeros((num_full_frames, num_instances, 3))
    instances_true_id = np.arange(num_instances)
    instances_model_types = np.ones(num_instances) * -1

    ego_to_world_start = np.loadtxt(
        os.path.join(datadir, "ego_pose", f"{selected_frames[0]:03d}.txt")
    )
    image_dir = os.path.join(datadir, 'images')
    n_cameras = len(cameras)
    n_images = len(os.listdir(image_dir))
    n_frames = n_images // n_cameras
    n_obj_in_frame = np.zeros(n_frames)
    for k, v in instances_info.items():
        int_k = int(k)
        objects_info[int_k] = dict()
        objects_info[int_k]['track_id'] = int_k
        objects_info[int_k]['class'] = v["class_name"]
        objects_info[int_k]['height'] = v['frame_annotations']['box_size'][int_k][2]
        objects_info[int_k]['width'] = v['frame_annotations']['box_size'][int_k][1]
        objects_info[int_k]['length'] = v['frame_annotations']['box_size'][int_k][0]
        for frame_id, obj_to_world, box_size in zip(v["frame_annotations"]["frame_idx"],
                                                    v["frame_annotations"]["obj_to_world"],
                                                    v["frame_annotations"]["box_size"]):

            n_obj_in_frame[frame_id] += 1
        #     [np.array(v[:2]).astype(np.float64), np.array([type]), np.array(v[4:]).astype(np.float64)]

        # tracklets_ls.append(tr_array)
        # instances_model_types[int(k)] = v["class_name"]


    max_obj_per_frame = int(n_obj_in_frame[start_frame:end_frame + 1].max())
    num_frames = end_frame - start_frame + 1
    visible_objects_ids = np.ones([num_frames, max_obj_per_frame]) * -1.0
    visible_objects_pose_vehicle = np.ones([num_frames, max_obj_per_frame, 7]) * -1.0
    visible_objects_pose_world = np.ones([num_frames, max_obj_per_frame, 7]) * -1.0

    for k, v in instances_info.items():
        int_k = int(k)
        for frame_id, obj_to_world, box_size, box_center_x, box_center_y, box_center_z, heading in zip(v["frame_annotations"]["frame_idx"],
                                                    v["frame_annotations"]["obj_to_world"],
                                                    v["frame_annotations"]["box_size"], v["frame_annotations"]["box_center_x"], v["frame_annotations"]["box_center_y"], v["frame_annotations"]["box_center_z"], v["frame_annotations"]["heading"]):
            if start_frame <= frame_id <= end_frame:
                # the first ego pose as the origin of the world coordinate system.
                obj_to_world = np.array(obj_to_world).reshape(4, 4)
                obj_to_world = np.linalg.inv(ego_to_world_start) @ obj_to_world
                instances_pose[frame_id, int(k)] = np.array(obj_to_world)
                instances_size[frame_id, int(k)] = np.array(box_size)

                ego_pose = ego_poses[frame_id]


                frame_idx = frame_id - start_frame
                obj_column = np.argwhere(visible_objects_ids[frame_idx, :] < 0).min()

                tr_array0 = frame_idx
                tr_array1 = int_k
                tr_array2 = v["class_name"]
                tr_array4 = box_size[2]
                tr_array5 = box_size[1]
                tr_array6 = box_size[0]
                tr_array7 = box_center_x
                tr_array8 = box_center_y
                tr_array9 = box_center_z
                tr_array10 = heading
                arrays = [np.array(tr_array7), np.array(tr_array8), np.array(tr_array9), np.array(tr_array10)]

                for i, arr in enumerate(arrays):
                    if arr.ndim == 0:
                        arrays[i] = np.array([arr])

                tracklet = np.concatenate(arrays)

                obj_pose_vehicle, obj_pose_world = make_obj_pose(ego_pose, tracklet)

                visible_objects_ids[frame_idx, obj_column] = int_k
                visible_objects_pose_vehicle[frame_idx, obj_column] = obj_pose_vehicle
                visible_objects_pose_world[frame_idx, obj_column] = obj_pose_world

    # get frame valid instances
    # shape (num_frames, num_instances)
    per_frame_instance_mask = np.zeros((num_full_frames, num_instances))
    for frame_idx, valid_instances in frame_instances.items():
        per_frame_instance_mask[int(frame_idx), valid_instances] = 1

    # select the frames that are in the range of start_timestep and end_timestep
    instances_pose = torch.from_numpy(instances_pose[selected_frames[0]:selected_frames[1]]).float()
    instances_size = torch.from_numpy(instances_size[selected_frames[0]:selected_frames[1]]).float()
    instances_true_id = torch.from_numpy(instances_true_id).long()
    instances_model_types = torch.from_numpy(instances_model_types).long()
    per_frame_instance_mask = torch.from_numpy(per_frame_instance_mask[selected_frames[0]:selected_frames[1]]).bool()

    # assign to the class

    # objects_info, tracklets_ls, n_obj_in_frame,

    box_scale = hyper.box_scale
    frames = list(range(start_frame, end_frame + 1))
    frames = np.array(frames).astype(np.int32)
    for key in objects_info.keys():
        obj = objects_info[key]
        if obj['class'] == 'pedestrian':
            obj['deformable'] = True
        else:
            obj['deformable'] = False
        obj['class_label'] = '1'
        obj['width'] = obj['width'] * box_scale
        obj['length'] = obj['length'] * box_scale
        obj_frame_idx = np.argwhere(visible_objects_ids == key)[:, 0]
        obj_frame_idx = obj_frame_idx.astype(np.int32)
        obj_frames = frames[obj_frame_idx]
        obj['start_frame'] = np.min(obj_frames)
        obj['end_frame'] = np.max(obj_frames)


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

    # , frame_timestamps[1]( padding)
    if max_timestamp > frame_timestamps[-1]:
        tracklets_last = tracklets[-1]
        frame_timestamps = np.concatenate([frame_timestamps, [max_timestamp]])
        tracklets = np.concatenate([tracklets, tracklets_last[None]], axis=0)

    return tracklets, frame_timestamps

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

def generate_kitti_dataparser_outputs(
        args, hyper,
        datadir,
        selected_frames=None,
        build_pointcloud=True,
        cameras=[0, 1]
):
    image_dir = os.path.join(datadir, 'images')
    image_filenames_all = sorted(glob(os.path.join(image_dir, '*.jpg')))
    num_frames_all = len(image_filenames_all) // 2
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
    from utils.colmap_kitti_full import run_colmap_kitti
    if build_pointcloud:
        print('Running COLMAP...')
        colmap_success = run_colmap_kitti(result, args)
        if not colmap_success:
            print("COLMAP processing failed. Using alternative initialization...")

            points = []
            colors = []


            for i in range(len(c2ws)):
                cam_pos = c2ws[i][:3, 3]

                num_points = 1000
                random_points = np.random.normal(cam_pos, scale=5.0, size=(num_points, 3))
                random_colors = np.random.randint(0, 256, size=(num_points, 3))

                points.append(random_points)
                colors.append(random_colors)

            points = np.vstack(points)
            colors = np.vstack(colors)

            # PLY
            ply_path = os.path.join(args.model_path, "input_ply/points3D_lidar.ply")
            os.makedirs(os.path.dirname(ply_path), exist_ok=True)
            storePly(ply_path, points, colors)
            print(f"Created alternative point cloud at {ply_path}")

    result['axis_transform'] = m_trans
    build_pointcloud = True
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
        points_colmap_path = os.path.join(args.model_path, 'colmap/triangulated/sparse/model/points3D.bin')
        points_colmap_xyz, points_colmap_rgb, points_colmap_error = read_points3D_binary(points_colmap_path)
        points_colmap_rgb = points_colmap_rgb / 255.

        print('initialize from lidar pointcloud')
        pointcloud_path = os.path.join(datadir, 'pointcloud.npz')
        print("pointcloud_path:", pointcloud_path)
        pd = np.load(pointcloud_path, allow_pickle=True)
        pts3d_dict = np.load(pointcloud_path, allow_pickle=True)['points'].item()
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

            # each point should be observed by at least one camera in camera lists
            mask = np.array([c in cameras for c in points_camera_all]).astype(np.bool_)

            # get filtered LiDAR pointcloud position and color
            points_xyz_vehicle = raw_3d[mask]

            # transfrom LiDAR pointcloud from vehicle frame to world frame
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

            # filer points in tracking bbox
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

        initial_num_obj = 20000

        for k, v in points_xyz_dict.items():
            if len(v) == 0:
                continue
            else:
                points_xyz = np.concatenate(v, axis=0)
                points_rgb = np.concatenate(points_rgb_dict[k], axis=0)
                if k == 'bkgd':
                    # downsample lidar pointcloud with voxels
                    points_lidar = o3d.geometry.PointCloud()
                    points_lidar.points = o3d.utility.Vector3dVector(points_xyz)
                    points_lidar.colors = o3d.utility.Vector3dVector(points_rgb)
                    downsample_points_lidar = points_lidar.voxel_down_sample(voxel_size=0.15)
                    downsample_points_lidar, _ = downsample_points_lidar.remove_radius_outlier(nb_points=10, radius=0.5)
                    points_lidar_xyz = np.asarray(downsample_points_lidar.points).astype(np.float32)
                    points_lidar_rgb = np.asarray(downsample_points_lidar.colors).astype(np.float32)
                elif k.startswith('obj'):
                    # points_obj.points = o3d.utility.Vector3dVector(points_xyz)
                    # points_obj.colors = o3d.utility.Vector3dVector(points_rgb)

                    if len(points_xyz) > initial_num_obj:
                        random_indices = np.random.choice(len(points_xyz), initial_num_obj, replace=False)
                        points_xyz = points_xyz[random_indices]
                        points_rgb = points_rgb[random_indices]

                    points_xyz_dict[k] = points_xyz
                    points_rgb_dict[k] = points_rgb

                else:
                    raise NotImplementedError()

        # Get sphere center and radius
        lidar_sphere_normalization = get_Sphere_Norm(points_lidar_xyz)
        sphere_center = lidar_sphere_normalization['center']
        sphere_radius = lidar_sphere_normalization['radius']

        # combine SfM pointcloud with LiDAR pointcloud
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
    return result


def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
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
