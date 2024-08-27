import torch
import numpy as np

from pydantic import BaseModel
from omegaconf import DictConfig

from core.geometry.gripper import Gripper

class VisualizationParams(BaseModel):
    visualize: bool = True
    visualize_detail: bool = False
    visualize_every_step: int = 1

class AlgoParams(BaseModel):
    # Setting Parameters
    total_num: int = 1024  # total number of pose sampled
    sample_time: int = 20  # number of pose sampled each run
    min_distance: float = 0.05  # minimum distance of initial pose to object surface
    max_distance: float = 0.05  # maximum distance of initial pose to object surface
    joint_barrier_ratio: float = 0.2  # barrier term ratio
    loop_1_num: int = 20  # iteration number of loop 1
    loop_2_num: int = 10  # iteration number of loop 2
    norm_filter: bool = True  # signal if filter the point-pair without opposite normal
    fc_point_num: bool = 4  # point number to compute force closure term
    collision_filter_threshold: int = 20

    ep_coeff: float = 2e0  # coefficient of Ep term
    en_coeff: float = 1e-3  # coefficient of En term
    efc_coeff: float = 1e-1  # coefficient of force closure term
    barrier_coeff: float = 0e-1  # coefficient of barrier term
    joints_limit_coeff: float = 2e0  # coefficient of joint limit term
    barrier_threshold: float = 0.1  # threshold distance of barrier term

    decay_rate: float = 1 - 1e-1  # decay rate of gradient update
    update_step_t: float = 2e-2  # update step of translation
    update_step_R: float = 1e-1  # update step of rotation
    update_step_q: float = 4e0  # update step of joint states

class AlgoConfig():

    def __init__(self,
                 visualization_params: DictConfig,
                 algo_params: DictConfig,
                 device: str):
        self.visualization_params = VisualizationParams(**visualization_params)
        self.algo_params = AlgoParams(**algo_params)
        self.device = device


        # Member Variables
        self.dtype = torch.float
        self.joint_init_value = None
        self.joints_limit_upper = None
        self.joints_limit_lower = None
        self.joints_limit_barrier = None
        self.link_partial_points = None
        self.link_partial_normals = None
        self.link_complete_points = None
        self.link_complete_normals = None
        self.sample_config = None
        self.dest_points = None
        self.dest_normals = None
        self.visualize_step_count = 0
        self.tmp_decay = 1




    def init_joint_param(self, gripper:Gripper):
        device = self.device
        joints = gripper.joints

        lower_tmp = np.zeros(len(joints))
        upper_tmp = np.zeros(len(joints))
        init_tmp = np.zeros(len(joints))

        for i in range(len(joints)):
            lower_tmp[i] = joints[i].limit.lower
            upper_tmp[i] = joints[i].limit.upper
            init_tmp[i] = 0.75 * lower_tmp[i] + 0.25 * upper_tmp[i]

        if gripper.name in ['shadow']:
            init_tmp[0] = 0.95 * lower_tmp[0] + 0.05 * upper_tmp[0]
            init_tmp[4] = 0.65 * lower_tmp[4] + 0.35 * upper_tmp[4]
            init_tmp[8] = 0.65 * lower_tmp[8] + 0.35 * upper_tmp[8]
            init_tmp[13] = 0.95 * lower_tmp[13] + 0.05 * upper_tmp[13]
            init_tmp[18] = 1
            init_tmp[17] = -1
            upper_tmp[0] = upper_tmp[4] = upper_tmp[8] = upper_tmp[13] = 0
        elif gripper.name in ['svh', 'svh_r']:
            init_tmp[0] = 0.9
        elif gripper.name in ['franka']:
            for i in range(len(joints)):
                init_tmp[i] = 0.15 * lower_tmp[i] + 0.85 * upper_tmp[i]

        self.joints_limit_lower = torch.from_numpy(lower_tmp).to(device).to(self.dtype)
        self.joints_limit_upper = torch.from_numpy(upper_tmp).to(device).to(self.dtype)
        self.joints_limit_barrier = self.algo_params.joint_barrier_ratio * torch.from_numpy(upper_tmp - lower_tmp).to(device).to(self.dtype)
        self.joint_init_value = torch.from_numpy(init_tmp).to(device).to(self.dtype)

    def init_sample_config(self, gripper:Gripper):
        if gripper.name in ['svh', 'svh_r']:
            self.sample_config = ['use_y2z']
        elif gripper.name in ['shadow']:
            self.sample_config = ['use_y2-z']
        elif gripper.name in ['barrett', 'franka']:
            self.sample_config = []
