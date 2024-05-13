# nuScenes dev-kit.
# Code written by Oscar Beijbom, 2018. Altered by Mehar Khurana, 2024.

import copy
import os.path as osp
import struct
from abc import ABC, abstractmethod
from functools import reduce
from typing import Tuple, List, Dict

import cv2
import numpy as np
from matplotlib.axes import Axes
from pyquaternion import Quaternion

from nuscenes.utils.geometry_utils import view_points, transform_matrix
from nuscenes.utils.splits import create_splits_scenes
from nuscenes import NuScenes
from nuscenes.map_expansion.map_api import NuScenesMap

import torch
import scipy

ATTRIBUTE_NAMES = {
    "barrier": "",
    "traffic_cone": "",
    "bicycle": "cycle.without_rider",
    "motorcycle": "cycle.without_rider",
    "pedestrian": "pedestrian.standing",
    "car": "vehicle.stopped",
    "bus": "vehicle.stopped",
    "construction_vehicle": "vehicle.stopped",
    "trailer": "vehicle.stopped",
    "truck": "vehicle.stopped",
    "trafficcone": "",
    "constructionvehicle": "vehicle.stopped",
}

CAM_LIST = ["CAM_FRONT", "CAM_FRONT_RIGHT", "CAM_BACK_RIGHT", "CAM_BACK", "CAM_BACK_LEFT", "CAM_FRONT_LEFT"]

class PointCloud(ABC):
    """
    Abstract class for manipulating and viewing point clouds.
    Every point cloud (lidar and radar) consists of points where:
    - Dimensions 0, 1, 2 represent x, y, z coordinates.
        These are modified when the point cloud is rotated or translated.
    - All other dimensions are optional. Hence these have to be manually modified if the reference frame changes.
    """

    def __init__(self, points: torch.Tensor):
        """
        Initialize a point cloud and check it has the correct dimensions.
        :param points: <np.float: d, n>. d-dimensional input point cloud matrix.
        """
        assert points.shape[0] == self.nbr_dims(), 'Error: Pointcloud points must have format: %d x n' % self.nbr_dims()
        self.points = points

    @staticmethod
    @abstractmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """
        pass

    @classmethod
    @abstractmethod
    def from_file(cls, file_name: str) -> 'PointCloud':
        """
        Loads point cloud from disk.
        :param file_name: Path of the pointcloud file on disk.
        :return: PointCloud instance.
        """
        pass

    @classmethod
    def from_file_multisweep(cls,
                             nusc: 'NuScenes',
                             sample_rec: Dict,
                             chan: str,
                             ref_chan: str,
                             nsweeps: int = 5,
                             min_distance: float = 1.0) -> Tuple['PointCloud', torch.Tensor]:
        """
        Return a point cloud that aggregates multiple sweeps.
        As every sweep is in a different coordinate frame, we need to map the coordinates to a single reference frame.
        As every sweep has a different timestamp, we need to account for that in the transformations and timestamps.
        :param nusc: A NuScenes instance.
        :param sample_rec: The current sample.
        :param chan: The lidar/radar channel from which we track back n sweeps to aggregate the point cloud.
        :param ref_chan: The reference channel of the current sample_rec that the point clouds are mapped to.
        :param nsweeps: Number of sweeps to aggregated.
        :param min_distance: Distance below which points are discarded.
        :return: (all_pc, all_times). The aggregated point cloud and timestamps.
        """
        # Init.
        points = torch.zeros((cls.nbr_dims(), 0), dtype=torch.float32 if cls == LidarPointCloud else torch.float64)
        all_pc = cls(points)
        all_times = torch.zeros((1, 0))

        # Get reference pose and timestamp.
        ref_sd_token = sample_rec['data'][ref_chan]
        ref_sd_rec = nusc.get('sample_data', ref_sd_token)
        ref_pose_rec = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
        ref_cs_rec = nusc.get('calibrated_sensor', ref_sd_rec['calibrated_sensor_token'])
        ref_time = 1e-6 * ref_sd_rec['timestamp']

        # Homogeneous transform from ego car frame to reference frame.
        ref_from_car = transform_matrix(ref_cs_rec['translation'], Quaternion(ref_cs_rec['rotation']), inverse=True)

        # Homogeneous transformation matrix from global to _current_ ego car frame.
        car_from_global = transform_matrix(ref_pose_rec['translation'], Quaternion(ref_pose_rec['rotation']),
                                           inverse=True)

        # Aggregate current and previous sweeps.
        sample_data_token = sample_rec['data'][chan]
        current_sd_rec = nusc.get('sample_data', sample_data_token)
        for _ in range(nsweeps):
            # Load up the pointcloud and remove points close to the sensor.
            current_pc = cls.from_file(osp.join(nusc.dataroot, current_sd_rec['filename']))
            current_pc.remove_close(min_distance)

            # Get past pose.
            current_pose_rec = nusc.get('ego_pose', current_sd_rec['ego_pose_token'])
            global_from_car = transform_matrix(current_pose_rec['translation'],
                                               Quaternion(current_pose_rec['rotation']), inverse=False)

            # Homogeneous transformation matrix from sensor coordinate frame to ego car frame.
            current_cs_rec = nusc.get('calibrated_sensor', current_sd_rec['calibrated_sensor_token'])
            car_from_current = transform_matrix(current_cs_rec['translation'], Quaternion(current_cs_rec['rotation']),
                                                inverse=False)

            # Fuse four transformation matrices into one and perform transform.
            trans_matrix = reduce(torch.matmul, [ref_from_car, car_from_global, global_from_car, car_from_current])
            current_pc.transform(trans_matrix)

            # Add time vector which can be used as a temporal feature.
            time_lag = ref_time - 1e-6 * current_sd_rec['timestamp']  # Positive difference.
            times = time_lag * torch.ones((1, current_pc.nbr_points()))
            all_times = torch.hstack((all_times, times))

            # Merge with key pc.
            all_pc.points = torch.hstack((all_pc.points, current_pc.points))

            # Abort if there are no previous sweeps.
            if current_sd_rec['prev'] == '':
                break
            else:
                current_sd_rec = nusc.get('sample_data', current_sd_rec['prev'])

        return all_pc, all_times

    def nbr_points(self) -> int:
        """
        Returns the number of points.
        :return: Number of points.
        """
        return self.points.shape[1]

    # def subsample(self, ratio: float) -> None:
    #     """
    #     Sub-samples the pointcloud.
    #     :param ratio: Fraction to keep.
    #     """
    #     selected_ind = np.random.choice(np.arange(0, self.nbr_points()), size=int(self.nbr_points() * ratio))
    #     self.points = self.points[:, selected_ind]

    def remove_close(self, radius: float) -> None:
        """
        Removes point too close within a certain radius from origin.
        :param radius: Radius below which points are removed.
        """

        x_filt = torch.abs(self.points[0, :]) < radius
        y_filt = torch.abs(self.points[1, :]) < radius
        not_close = torch.logical_not(torch.logical_and(x_filt, y_filt))
        self.points = self.points[:, not_close]

    def translate(self, x: torch.Tensor) -> None:
        """
        Applies a translation to the point cloud.
        :param x: <torch.float: 3, 1>. Translation in x, y, z.
        """
        for i in range(3):
            self.points[i, :] = self.points[i, :] + x[i]

    def rotate(self, rot_matrix: torch.Tensor) -> None:
        """
        Applies a rotation.
        :param rot_matrix: <torch.float: 3, 3>. Rotation matrix.
        """
        self.points[:3, :] = torch.matmul(rot_matrix, self.points[:3, :])

    def transform(self, transf_matrix: torch.Tensor) -> None:
        """
        Applies a homogeneous transform.
        :param transf_matrix: <torch.float: 4, 4>. Homogenous transformation matrix.
        """
        self.points[:3, :] = transf_matrix.matmul(torch.vstack((self.points[:3, :], torch.ones(self.nbr_points()))))[:3, :]

    def render_height(self,
                      ax: Axes,
                      view: torch.Tensor = torch.eye(4),
                      x_lim: Tuple[float, float] = (-20, 20),
                      y_lim: Tuple[float, float] = (-20, 20),
                      marker_size: float = 1) -> None:
        """
        Very simple method that applies a transformation and then scatter plots the points colored by height (z-value).
        :param ax: Axes on which to render the points.
        :param view: <torch.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max). x range for plotting.
        :param y_lim: (min, max). y range for plotting.
        :param marker_size: Marker size.
        """
        self._render_helper(2, ax, view, x_lim, y_lim, marker_size)

    def render_intensity(self,
                         ax: Axes,
                         view: torch.Tensor = torch.eye(4),
                         x_lim: Tuple[float, float] = (-20, 20),
                         y_lim: Tuple[float, float] = (-20, 20),
                         marker_size: float = 1) -> None:
        """
        Very simple method that applies a transformation and then scatter plots the points colored by intensity.
        :param ax: Axes on which to render the points.
        :param view: <torch.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max).
        :param y_lim: (min, max).
        :param marker_size: Marker size.
        """
        self._render_helper(3, ax, view, x_lim, y_lim, marker_size)

    def _render_helper(self,
                       color_channel: int,
                       ax: Axes,
                       view: torch.Tensor,
                       x_lim: Tuple[float, float],
                       y_lim: Tuple[float, float],
                       marker_size: float) -> None:
        """
        Helper function for rendering.
        :param color_channel: Point channel to use as color.
        :param ax: Axes on which to render the points.
        :param view: <torch.float: n, n>. Defines an arbitrary projection (n <= 4).
        :param x_lim: (min, max).
        :param y_lim: (min, max).
        :param marker_size: Marker size.
        """
        points = view_points(self.points[:3, :], view, normalize=False)
        ax.scatter(points[0, :], points[1, :], c=self.points[color_channel, :], s=marker_size)
        ax.set_xlim(x_lim[0], x_lim[1])
        ax.set_ylim(y_lim[0], y_lim[1])


class LidarPointCloud(PointCloud):

    @staticmethod
    def nbr_dims() -> int:
        """
        Returns the number of dimensions.
        :return: Number of dimensions.
        """
        return 4

    @classmethod
    def from_file(cls, file_name: str, device) -> 'LidarPointCloud':
        """
        Loads LIDAR data from binary numpy format. Data is stored as (x, y, z, intensity, ring index).
        :param file_name: Path of the pointcloud file on disk.
        :return: LidarPointCloud instance (x, y, z, intensity).
        """

        assert file_name.endswith('.bin'), 'Unsupported filetype {}'.format(file_name)

        scan = torch.from_numpy(np.fromfile(file_name, dtype=np.float32)).to(device=device)
        points = scan.reshape((-1, 5))[:, :cls.nbr_dims()]
        return cls(points.T)



# nuScenes dev-kit.
# Code written by Oscar Beijbom and Alex Lang, 2018.
def view_points(points: torch.Tensor,
    view: torch.Tensor,
    normalize: bool,
    device: str) -> (torch.Tensor, torch.Tensor):
    """
    This function transforms 3D points in global frame to a camera view.
    :param points: <torch.float: 3, n>. Input point cloud matrix.
    :param view: <torch.float: n, n>. Defines an arbitrary projection.
    :param normalize: Whether to normalize the points.
    :param device: Device to run on.
    :return: <torch.float: 3, n>. Returns the transformed points.
    """
    
    assert view.shape[0] <= 4
    assert view.shape[1] <= 4
    assert points.shape[0] == 3

    viewpad = torch.eye(4).to(device=device, dtype=torch.float32)
    viewpad[:view.shape[0], :view.shape[1]] = view

    nbr_points = points.shape[1]

    # Do operation in homogenous coordinates.
    points = torch.concatenate((points, torch.ones((1, nbr_points)).to(device=device, dtype=torch.float32)))
    points = torch.matmul(viewpad, points)

    points = points[:3, :]
    point_depths = torch.clone(points[2, :])

    if normalize:
        points = points / points[2:3, :].repeat(3, 1).reshape(3, nbr_points)

    return points, point_depths



# nuscenes utils
def count_frames(nusc: NuScenes,
    sample: str
    ):
    frame_count = 1

    if sample["next"] != "":
        frame_count += 1

        # Don't want to change where sample['next'] points to since it's used later, so we'll create our own pointer
        sample_counter = nusc.get('sample', sample['next'])

        while sample_counter['next'] != '':
            frame_count += 1
            sample_counter = nusc.get('sample', sample_counter['next'])
    
    return frame_count

def get_shape_prior(
    shape_priors: Dict[str, List[float]],
    name: str,
    chatgpt: bool = False):
    # add priors for more categories
    if not chatgpt:
        if name == "car":
            return shape_priors["vehicle.car"]
        elif name == "bicycle":
            return shape_priors["vehicle.bicycle"]
        elif name == "bus":
            return shape_priors["vehicle.bus.rigid"]
        elif name == "truck":
            return shape_priors["vehicle.truck"]
        elif name == "pedestrian":
            return shape_priors["human.pedestrian.adult"]
        elif name == "trafficcone":
            return shape_priors["movable_object.trafficcone"]
        elif name == "constructionvehicle":
            return shape_priors["vehicle.construction"]
        elif name == "motorcycle":
            return shape_priors["vehicle.motorcycle"]
        elif name == "trailer":
            return shape_priors["vehicle.trailer"]
        elif name == "child":
            return shape_priors["human.pedestrian.child"]
        elif name == "stroller":
            return shape_priors["human.pedestrian.adult"]
        elif name == "barrier":
            return shape_priors["movable_object.barrier"]
        
    
    else:
        return shape_priors[name]


def get_nusc_map(
    nusc: NuScenes,
    scene: Dict,
    INPUT_PATH: str
    ) -> NuScenesMap:

    # Get scene location
    log = nusc.get("log", scene["log_token"])
    location = log["location"]

    # Get nusc map
    nusc_map = NuScenesMap(dataroot=INPUT_PATH, map_name=location)

    return nusc_map


def get_all_lane_points_in_scene(
    nusc_map: NuScenesMap
    ) -> Tuple[Dict[str, List[List[float]]], List[List[float]]]:

    # Aggregate all lanes and lane connectors
    lane_records = nusc_map.lane + nusc_map.lane_connector
    lane_tokens = [lane["token"] for lane in lane_records]

    # Get all lane points
    lane_pt_dict = nusc_map.discretize_lanes(lane_tokens, 0.5) # 0.5m discretization

    # Aggregate all lane points
    all_lane_pts = []
    for lane_pts in lane_pt_dict.values():
        for lane_pt in lane_pts:
            all_lane_pts.append(lane_pt)
    
    return lane_pt_dict, all_lane_pts


def distance_matrix_lanes(
    A: torch.Tensor,
    B: torch.Tensor,
    squared: bool = False
    ) -> torch.Tensor:

    A = A.to(device=DEVICE)
    B = B.to(device=DEVICE)

    M = A.shape[0]
    N = B.shape[0]

    assert A.shape[1] == B.shape[1], f"The number of components for vectors in A \
        {A.shape[1]} does not match that of B {B.shape[1]}!"

    A_dots = torch.mul(
                torch.mul(A, A).sum(dim=1).reshape((M,1)),
                torch.Tensor(np.ones(shape=(1,N))).to(device=DEVICE)
            )
    B_dots = torch.mul(
                torch.mul(B, B).sum(dim=1),
                torch.Tensor(np.ones(shape=(M,1))).to(device=DEVICE)
            )
    D_squared =  torch.sub(
                    torch.add(A_dots, B_dots),
                    torch.mul(
                        2, torch.mm(
                            A, B.transpose(0, 1)
                        )
                    )
                )

    if squared == False:
        zero_mask = torch.less(D_squared, 0.0)
        D_squared[zero_mask] = 0.0
        return torch.sqrt(D_squared)

    return D_squared


def lane_yaws_distances_and_coords(
    all_centroids: List[List[float]],
    all_lane_pts: List[List[float]],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:

    all_lane_pts = torch.Tensor(all_lane_pts).to(device='cpu')
    all_centroids = torch.Tensor(all_centroids).to(device='cpu')
    # print(all_lane_pts, all_centroids)
    # start = time.time()
    # DistMat = distance_matrix_lanes(all_centroids[:, :2], all_lane_pts[:, :2])
    # DistMat = distance_matrix(all_centroids[:, :2], all_lane_pts[:, :2])
    DistMat = scipy.spatial.distance.cdist(all_centroids[:, :2], all_lane_pts[:, :2])
    
    min_lane_indices = np.argmin(DistMat, axis=1)
    # print(min_lane_indices)
    distances = np.min(DistMat, axis=1)

    all_lane_pts = np.array(all_lane_pts)
    min_lanes = np.array([all_lane_pts[min_lane_indices[0]]])
    for idx in min_lane_indices:
        # print(idx)
        min_lanes = np.vstack([min_lanes, all_lane_pts[idx, :]])
    
    # print(min_lanes.shape)

    yaws = min_lanes[1:, 2]
    coords = min_lanes[1:, :2]

    # print(distances.shape, yaws.shape, coords.shape)
    # end = time.time()

    # print(f"Closest lane took {end - start} seconds.")
    # timer['closest lane'] += end - start

    return yaws, distances, coords