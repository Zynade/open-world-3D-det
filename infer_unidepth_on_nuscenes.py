from unidepth.models import UniDepthV1

import numpy as np
from PIL import Image

from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion
from nuscenes.utils.data_classes import Quaternion as nQuaternion
from utils.nuscenes import LidarPointCloud, view_points, count_frames, get_shape_prior, ATTRIBUTE_NAMES, CAM_LIST
from utils.nuscenes import lane_yaws_distances_and_coords, distance_matrix_lanes, get_all_lane_points_in_scene
from utils.nuscenes import get_nusc_map
from utils.utils import get_medoid
import os
from pathlib import Path
from pycocotools import mask as mask_utils
import json
import numpy as np
import cv2
import PIL
from tqdm import tqdm
from typing import List
import torch
from PCADetection import PCADet

# declare root dir global
ROOT = Path(__file__).resolve().parents[0]

# NUSCENES_DATA = "datasets/nuScenes-mini"
NUSCENES_DATA = "/data2/mehark/nuScenes/nuScenes/"
NUSCENES_OUTPUT = "outputs/nuScenes-mini/"
NUSCENES_VERSION = "v1.0-trainval"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CAM_LIST = ['CAM_FRONT', 'CAM_FRONT_LEFT', 'CAM_FRONT_RIGHT', 'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT']


def infer_image(image_path, intrinsics, model):
    # Load the RGB image and the normalization will be taken care of by the model
    rgb = torch.from_numpy(np.array(Image.open(image_path))).permute(2, 0, 1) # C, H, W
    # # print(intrinsics)
    # predictions = model.infer(rgb)

    # # Metric Depth Estimation
    # depth = predictions["depth"]
    # print(depth.shape)
    # print(depth)
    # xyz = predictions["points"]
    # pred_intrinsics = predictions["intrinsics"]

    predictions = model.infer(rgb, intrinsics)

    # Metric Depth Estimation
    depth = predictions["depth"]
    xyz = predictions["points"]

    return xyz, depth


def main():

    # Load validation set
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATA, verbose=True)
    nusc.list_scenes()
    minival_scenes = ['scene-0103', 'scene-0916']

    # Move to CUDA, if any
    model = UniDepthV1.from_pretrained("lpiccinelli/unidepth-v1-vitl14")
    device = torch.device(DEVICE)
    model = model.to(device)

    for scene in tqdm(nusc.scene):
        if not scene['name'] in minival_scenes:
            continue
        
        sample_token = scene['first_sample_token']

        sample = nusc.get('sample', sample_token)

        while sample['next'] != '':
            for cam in CAM_LIST:
                # Get the camera sensor
                camera = nusc.get('sample_data', sample['data'][cam])

                # Get intrinsics
                camera_intrinsic = torch.from_numpy(np.array(nusc.get('calibrated_sensor', camera['calibrated_sensor_token'])['camera_intrinsic'])).float()

                # Get the image path
                image_path = nusc.get_sample_data_path(camera['token'])

                # Infer the image
                xyz, depth = infer_image(image_path, camera_intrinsic, model)

                xyz = xyz.cpu().numpy()
                # XYZ Path to save
                xyz_path = os.path.join(NUSCENES_OUTPUT, 'samples-pseudodepth', cam, image_path.split("/")[-1].replace("jpg", "xyz"))
                os.makedirs(os.path.dirname(xyz_path), exist_ok=True)
                np.save(xyz_path, xyz)

            sample = nusc.get('sample', sample['next'])


if __name__ == "__main__":
    main()