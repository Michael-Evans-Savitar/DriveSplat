#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
import os
import random
import json
import torch
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, storePly
from scene.gaussian_model import GaussianModel
from scene.gaussian_model_all import DriveSplatModel
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON
from scene.dataset import Dataset
from typing import Union

class Scene:

    # gaussians : GaussianModel
    gaussians : Union[GaussianModel, DriveSplatModel]
    dataset: Dataset
    def __init__(self, args : ModelParams, dataset: Dataset, gaussians : Union[GaussianModel, DriveSplatModel], load_iteration=None, shuffle=True, resolution_scales=[1.0], ply_path=None, logger=None, is_training=True):
        """
        :param path: Path to colmap scene main folder.
        """
        self.gaussians = gaussians
        self.dataset = dataset
        self.model_path = args.model_path
        self.loaded_iter = None
        self.resolution_scales = resolution_scales

        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration

            print("Loading trained model at iteration {}".format(self.loaded_iter))

        self.train_cameras = {}
        self.test_cameras = {}

        scene_info = dataset.scene_info

        self.gaussians.set_appearance(len(scene_info.train_cameras))
        if not self.loaded_iter:
            points = self.save_ply(scene_info.point_cloud, args.ratio, os.path.join(self.model_path, "input.ply"))
            json_cams = []
            camlist = []
            if scene_info.test_cameras:
                camlist.extend(scene_info.test_cameras)
            if scene_info.train_cameras:
                camlist.extend(scene_info.train_cameras)
            for id, cam in enumerate(camlist):
                json_cams.append(camera_to_JSON(id, cam))
            with open(os.path.join(self.model_path, "cameras.json"), 'w') as file:
                json.dump(json_cams, file)

        if shuffle:
            random.shuffle(scene_info.train_cameras)  # Multi-res consistent random shuffling
            random.shuffle(scene_info.test_cameras)  # Multi-res consistent random shuffling

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        for resolution_scale in self.resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
        if self.loaded_iter:
            # Resume training from a saved model still needs LOD metadata such as
            # init_level/levels before set_coarse_interval() is called.
            points = torch.tensor(scene_info.point_cloud.points[::args.ratio]).float().cuda()
            points = torch.unique(points, dim=0)
            self.gaussians.set_level(points, self.train_cameras, self.resolution_scales, args.dist_ratio, args.init_level, args.levels, args.model_path)
            self.gaussians.load_ply_sparse_gaussian(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
            self.gaussians.load_mlp_checkpoints(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter)))
            self.gaussians.load_deform_weights(self.model_path, self.loaded_iter)
            actor_pose_path = os.path.join(
                self.model_path, "point_cloud", f"iteration_{self.loaded_iter}", "actor_pose.pth"
            )
            if hasattr(self.gaussians, "actor_pose") and self.gaussians.actor_pose is not None:
                if os.path.exists(actor_pose_path):
                    try:
                        ap_state = torch.load(actor_pose_path, map_location="cuda")
                        self.gaussians.actor_pose.load_state_dict(ap_state)
                        print(f"[Scene]  Loaded actor_pose from {actor_pose_path}")
                    except Exception as e:
                        print(f"[Scene]  Failed to load actor_pose from {actor_pose_path}: {e}")
                else:
                    print(f"[Scene]  actor_pose not found: {actor_pose_path} (this will hurt foreground quality if opt_track was used)")
        else:
            if args.random_background:
                logger.info("Using random background")
            elif args.white_background:
                logger.info("Using white background")
            else:
                logger.info("Using black background")
            points = torch.unique(points, dim=0) # remove duplicate points
            self.gaussians.set_level(points, self.train_cameras, self.resolution_scales, args.dist_ratio, args.init_level, args.levels, args.model_path)
            input_path = args.model_path
            first_cam_center = None
            if len(scene_info.train_cameras) > 0:
                first_cam = self.train_cameras[1.0][0]
                first_cam_center = first_cam.camera_center.clone().detach().to(device="cuda")
            self.gaussians.create_from_pcd(points, self.cameras_extent, first_cam_center, logger, input_path)

    def save_ply(self, pcd, ratio, path):
        points = torch.tensor(pcd.points[::ratio]).float().cuda()
        colors = torch.tensor(pcd.colors[::ratio]).float().cuda()
        storePly(path, points.cpu().numpy(), colors.cpu().numpy())
        return points

    def save(self, iteration, stage):
        if stage == "coarse":
            point_cloud_path = os.path.join(self.model_path, "point_cloud/coarse_iteration_{}".format(iteration))

        else:
            point_cloud_path = os.path.join(self.model_path, "point_cloud/iteration_{}".format(iteration))
        self.gaussians.save_ply(os.path.join(point_cloud_path, "point_cloud.ply"))
        self.gaussians.save_mlp_checkpoints(point_cloud_path)
        self.gaussians.save_deform_weights(self.model_path, iteration)
        if hasattr(self.gaussians, "actor_pose") and self.gaussians.actor_pose is not None:
            try:
                actor_pose_path = os.path.join(point_cloud_path, "actor_pose.pth")
                torch.save(self.gaussians.actor_pose.save_state_dict(is_final=True), actor_pose_path)
                print(f"[Scene]  Saved actor_pose to {actor_pose_path}")
            except Exception as e:
                print(f"[Scene]  Failed to save actor_pose: {e}")

    def getTrainCameras(self):
        all_cams = []
        for scale in self.resolution_scales:
            all_cams.extend(self.train_cameras[scale])
        return all_cams

    def getTestCameras(self):
        all_cams = []
        for scale in self.resolution_scales:
            all_cams.extend(self.test_cameras[scale])
        return all_cams
