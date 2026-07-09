import os
import random
import json
import torch
from utils.system_utils import searchForMaxIteration
from scene.dataset_readers import sceneLoadTypeCallbacks, storePly
from arguments import ModelParams
from utils.camera_utils import cameraList_from_camInfos, camera_to_JSON

class Dataset():

    def __init__(self, args : ModelParams, hyper, load_iteration=None, shuffle=True, resolution_scales=[1.0], ply_path=None, logger=None):
        """
        :param path: Path to colmap scene main folder.
        """
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
        if os.path.exists(os.path.join(args.source_path, "sparse")):
            scene_info = sceneLoadTypeCallbacks["Colmap"](args, hyper, args.source_path, args.images, args.eval, args.ds)
        elif os.path.exists(os.path.join(args.source_path, "track")):
            scene_info = sceneLoadTypeCallbacks["Waymo"](args, hyper, args.source_path, args.images, args.eval, args.ds)
        elif os.path.exists(os.path.join(args.source_path, "instances")):
            scene_info = sceneLoadTypeCallbacks["KITTI"](args, hyper, args.source_path, args.images, args.eval, args.ds)

        self.cameras_extent = scene_info.nerf_normalization["radius"]

        self.scene_info = scene_info
