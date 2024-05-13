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

# declare root dir global
ROOT = Path(__file__).resolve().parents[0]

# NUSCENES_DATA = "datasets/nuScenes-mini"
NUSCENES_DATA = "/data2/mehark/nuScenes/nuScenes/"
NUSCENES_OUTPUT = "outputs/nuScenes-mini/"
NUSCENES_VERSION = "v1.0-trainval"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
min_dist= 2.3
shape_priors = json.load(open("outputs/nuScenes-mini/shape_priors.json"))





def create_bboxes_nuscenes(nusc: NuScenes, val_scenes: List[str], save_vis: bool = False,
    use_lanes_for_orientation: bool = False):
    """
    Create bounding boxes for the NuScenes dataset, using masked lidar points.
    
    Args:
        nusc (NuScenes): Instance of NuScenes dataset.
        val_scenes (list): List of scene names to process.
        save_vis (bool): Whether to save visualizations of predictions. Default is False.
    """
    
    predictions = {
        "meta": {
            "use_camera": True,
            "use_lidar": False,
            "use_radar": False,
            "use_map": True,
            "use_external": False,
        },
        "results": {}
    }

    json_save_path = ROOT / NUSCENES_OUTPUT / "mask_results_preds.json"
    with open(json_save_path, "r") as f:
        masks_json = json.load(f)
    

    # Scene-level loop
    for scene in tqdm(nusc.scene):
        if not scene['name'] in val_scenes:
            continue
        
        my_scene = scene
        sample_token = my_scene['first_sample_token']

        sample = nusc.get('sample', sample_token)

        # Compute number of frames for progress bar
        num_frames = count_frames(nusc, sample)
        progress_bar = tqdm(range(num_frames))

        # Initialize lists to keep track of centroids their ids
        all_centroids_list = []
        centroid_ids = []
        id_offset = -1
        id_offset_list1 = []

        # Frame-level loop
        for frame_num in progress_bar:

            cam_data_dict = {}
            for camera in CAM_LIST:
                camera_token = sample['data'][camera]

                # Here we just grab the front camera
                cam_data = nusc.get('sample_data', camera_token)
                poserecord = nusc.get('ego_pose', cam_data['ego_pose_token'])
                cs_record = nusc.get('calibrated_sensor', cam_data['calibrated_sensor_token'])

                data = {
                    "ego_pose": poserecord,
                    "calibrated_sensor": cs_record,
                }

                cam_data_dict[camera] = data

            pointsensor_token = sample['data']["LIDAR_TOP"]
            pointsensor = nusc.get('sample_data', pointsensor_token)
            
            aggr_set = []
            pointsensor_next = nusc.get('sample_data', pointsensor_token)

            # Loop for LiDAR pcd aggregation
            for i in range(3):
                pcl_path = os.path.join(nusc.dataroot, pointsensor_next['filename'])
                pc = LidarPointCloud.from_file(pcl_path, DEVICE)

                lidar_points = pc.points
                mask = torch.ones(lidar_points.shape[1]).to(device=DEVICE)
                mask = torch.logical_and(mask, torch.abs(lidar_points[0, :]) < np.sqrt(min_dist))
                mask = torch.logical_and(mask, torch.abs(lidar_points[1, :]) < np.sqrt(min_dist))
                lidar_points = lidar_points[:, ~mask]
                pc = LidarPointCloud(lidar_points)

                # Points live in the point sensor frame. So they need to be transformed via global to the image plane.
                # First step: transform the pointcloud to the ego vehicle frame for the timestamp of the sweep.
                cs_record = nusc.get('calibrated_sensor', pointsensor_next['calibrated_sensor_token'])
                pc.rotate(torch.from_numpy(Quaternion(cs_record['rotation']).rotation_matrix).to(device=DEVICE, dtype=torch.float32))
                pc.translate(torch.from_numpy(np.array(cs_record['translation'])).to(device=DEVICE, dtype=torch.float32))

                # Second step: transform from ego to the global frame.
                poserecord = nusc.get('ego_pose', pointsensor_next['ego_pose_token'])
                pc.rotate(torch.from_numpy(Quaternion(poserecord['rotation']).rotation_matrix).to(device=DEVICE, dtype=torch.float32))
                pc.translate(torch.from_numpy(np.array(poserecord['translation'])).to(device=DEVICE, dtype=torch.float32))

                aggr_set.append(pc.points)
                try:
                    pointsensor_next = nusc.get('sample_data', pointsensor_next['next'])
                except KeyError:
                    break
            
            aggr_pc_points = torch.hstack(tuple([pcd for pcd in aggr_set]))
            print(aggr_pc_points.shape)

            # Get the masks for this frame
            for mask_dict in masks_json:
                id_offset += 1

                mask_sample_token = nusc.get('sample_data', mask_dict["sample_data_token"])["sample_token"]
                mask_sample_data = nusc.get('sample_data', mask_dict["sample_data_token"])
                mask_channel = mask_sample_data["channel"]
                # print(mask_sample_token, sample_token, mask_channel, mask_dict["category"])
                if mask_sample_token != sample["token"]:
                    continue
                if mask_dict["score"] < 0.1:
                    continue

                print(mask_channel, mask_dict["category"], mask_dict["score"])
                # import time; time.sleep(0.5)


                image_size = mask_dict["mask"]["size"]
                mask_rle = mask_dict["mask"] # RLE encoded mask
                mask_array = mask_utils.decode(mask_rle)

                cam_data = cam_data_dict[mask_channel]

                mask_1 = PIL.Image.fromarray(mask_array)
                mask_array = mask_array[:, :].astype(bool)
                mask_array = torch.transpose(torch.from_numpy(mask_array).to(device=DEVICE, dtype=bool), 1, 0)

                track_points = np.array(range(aggr_pc_points.shape[1]))

                # pass in a copy of the aggregate pointcloud array
                # reset the lidar pointcloud
                cam_pc = LidarPointCloud(torch.clone(aggr_pc_points))

                # transform from global into the ego vehicle frame for the timestamp of the image.
                # poserecord = nusc.get('ego_pose', cam['ego_pose_token'])
                poserecord = cam_data['ego_pose']
                cam_pc.translate(torch.from_numpy(-np.array(poserecord['translation'])).to(device=DEVICE, dtype=torch.float32))
                cam_pc.rotate(torch.from_numpy(Quaternion(poserecord['rotation']).rotation_matrix.T).to(device=DEVICE, dtype=torch.float32))

                # transform from ego into the camera.
                # cs_record = nusc.get('calibrated_sensor', cam['calibrated_sensor_token'])
                cs_record = cam_data['calibrated_sensor']
                cam_pc.translate(torch.from_numpy(-np.array(cs_record['translation'])).to(device=DEVICE, dtype=torch.float32))
                cam_pc.rotate(torch.from_numpy(Quaternion(cs_record['rotation']).rotation_matrix.T).to(device=DEVICE, dtype=torch.float32))

                depths = cam_pc.points[2, :]

                # fetch the camera intrinsics
                camera_intrinsic = torch.from_numpy(np.array(cs_record["camera_intrinsic"])).to(device=DEVICE, dtype=torch.float32)
                # camera_intrinsic = camera_intrinsic*ratio
                camera_intrinsic[2, 2] = 1

                # Take the actual picture (matrix multiplication with camera-matrix + renormalization).
                points, point_depths = view_points(cam_pc.points[:3, :], camera_intrinsic, normalize=True, device=DEVICE)

                image_mask = mask_array # (W, H)
                # Create a boolean mask where True corresponds to masked pixels
                masked_pixels = (image_mask == 1) # (W, H)

                # Use np.logical_and to find points within masked pixels
                points_within_image = torch.logical_and(torch.logical_and(torch.logical_and(torch.logical_and(
                    depths > min_dist,                      # depths (N)
                    points[0] > 0),                          # points (3, N) -> points[0, :] (1, N)
                    points[0] < image_mask.shape[0] - 1),    # ^
                    points[1] > 0),                          # ^
                    points[1] < image_mask.shape[1] - 1     # ^
                )

                # floor points to get integer indices
                floored_points = torch.floor(points[:, points_within_image]).to(dtype=int) # (N_masked,)
                track_points = track_points[points_within_image.cpu()]

                points_within_mask = torch.logical_and(
                    floored_points,
                    masked_pixels[floored_points[0], floored_points[1]]
                )

                indices_within_mask = torch.where(torch.logical_and(torch.logical_and(points_within_mask[0, :], points_within_mask[1, :]), points_within_mask[2, :]))[0]
                masked_points_pixels = torch.where(points_within_mask)

                track_points = track_points[indices_within_mask.cpu()]

                global_masked_points = aggr_pc_points[:, track_points]

                if global_masked_points.numel() == 0:
                    continue

                id_offset_list1.append(id_offset)

                if len(global_masked_points.shape) == 1:
                    global_masked_points = torch.unsqueeze(global_masked_points, 1)
                global_centroid = get_medoid(global_masked_points[:3, :].to(dtype=torch.float32, device=DEVICE))
                
                mask_pc = LidarPointCloud(global_masked_points[:, global_centroid][None].T)

                centroid = mask_pc.points[:3]
                all_centroids_list.append(torch.Tensor(centroid).to(DEVICE, dtype=torch.float32))
                centroid_ids.append(id_offset)
                final_id_offset = id_offset

                print(centroid)
                print("-"*50)
            
            print("\n"*50)

            if sample['next'] != "":
                sample = nusc.get('sample', sample['next'])


        sample = nusc.get('sample', scene['first_sample_token'])
        id_offset = -1
        id_offset_list2 = []

        all_centroids_list = torch.stack(all_centroids_list)
        all_centroids_list = torch.squeeze(all_centroids_list)

        if use_lanes_for_orientation:
            # Get the map for this scene
            nusc_map = get_nusc_map(nusc, scene)

            # Get all lane objects and list of lane points
            lane_pt_dict, lane_pt_list = get_all_lane_points_in_scene(nusc_map)

            yaw_list, min_distance_list, lane_pt_coords_list = lane_yaws_distances_and_coords(
                all_centroids_list, lane_pt_list
            )

        for frame_num in range(num_frames):
            predictions["results"][sample["token"]] = []
            
            for mask_dict in masks_json:
                id_offset += 1
                if id_offset not in centroid_ids:
                    continue
                else:
                    # print("id_offset:", id_offset)
                    id = centroid_ids.index(id_offset)
                    final_id_offset2 = id_offset
                
                id_offset_list2.append(id_offset)

                category = mask_dict["category"]
                score = mask_dict["score"]
                centroid = np.squeeze(np.array(all_centroids_list[id, :].to(device='cpu')))
                m_x, m_y, m_z = [float(i) for i in centroid]

                extents = get_shape_prior(shape_priors, category)

                if use_lanes_for_orientation:
                    lane_yaw = yaw_list[id]
                    # dist_from_lane = min_distance_list[id]
                    LANE_ALIGNED = ["car", "truck", "bus", "construction_vehicle", "trailer", "barrier"]
                else:
                    LANE_ALIGNED = []
                

                if category in LANE_ALIGNED:
                    align_mat = np.eye(3)
                    align_mat[0:2, 0:2] = [[np.cos(lane_yaw), -np.sin(lane_yaw)], [np.sin(lane_yaw), np.cos(lane_yaw)]]

                else:
                    align_mat = np.eye(3)

                rot_quaternion = Quaternion(matrix=align_mat)

                if category == 'trafficcone':
                    category = 'traffic_cone'
                elif category == 'constructionvehicle':
                    category = 'construction_vehicle'

                box_dict = {
                        "sample_token": sample["token"],
                        "translation": [float(i) for i in centroid],
                        "size": list(extents),
                        "rotation": list(rot_quaternion),
                        "velocity": [0, 0],
                        "detection_name": category,
                        "detection_score": score,
                        "attribute_name": ATTRIBUTE_NAMES[category]
                    }

                assert sample["token"] in predictions["results"]

                predictions["results"][sample["token"]].append(box_dict)

            if sample['next'] != "":
                sample = nusc.get('sample', sample['next'])

    with open(os.path.join(NUSCENES_OUTPUT, "predictions_naive.json"), "w") as f:
        json.dump(predictions, f)


def main():

    # Load validation set
    nusc = NuScenes(version=NUSCENES_VERSION, dataroot=NUSCENES_DATA, verbose=True)
    nusc.list_scenes()
    minival_scenes = ['scene-0103', 'scene-0916']

    create_bboxes_nuscenes(nusc, minival_scenes, use_lanes_for_orientation=True)


if __name__ == "__main__":
    main()
