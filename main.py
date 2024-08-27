import os
import sys
import json
import torch
import hydra
import trimesh
import numpy as np
import open3d as o3d
import os.path as osp

from tqdm import trange
from typing import Tuple
from omegaconf import DictConfig

sys.path.append('.')
from utils import visualize_point_cloud, load_obj, export_meshes, rfu_filter
from core.config import AlgoConfig
from core.geometry.gripper import Gripper
from core.geometry.samplePosition import sample_position
from core.collision.graspCollisionCheckAvoidCollision import grasp_collision_check_avoid_collision
from core.dipgrasp import dipgrasp

torch.set_default_dtype(torch.float)

def generate_grasp_for_obj(point_cloud: o3d.geometry.PointCloud, gripper: Gripper, cfg: AlgoConfig, sample_num: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    if sample_num > 0:
        cfg.total_num = sample_num

    device = cfg.device
    Tbase = torch.eye(4).to(device)
    src_points, src_normals, src_weights = gripper.compute_pcd(Tbase.unsqueeze(0),
                                                               cfg.joint_init_value.unsqueeze(0),
                                                               cfg.link_partial_points,
                                                               cfg.link_partial_normals,
                                                               True)

    dest_points = torch.from_numpy(np.array(point_cloud.points))
    dest_normals = torch.from_numpy(np.array(point_cloud.normals))

    if dest_points.shape[0] > cfg.algo_params.total_num:
        downsample_idx = torch.randperm(dest_points.shape[0])[:cfg.total_num]
        dest_points = dest_points[downsample_idx]
        dest_normals = dest_normals[downsample_idx]

    points = dest_points.numpy()
    normals = dest_normals.numpy()

    idx = 0
    idx_list = [0]
    while idx < points.shape[0]:
        idx += cfg.algo_params.sample_time
        if idx > points.shape[0]:
            idx = points.shape[0]
        idx_list.append(idx)
    pose_list = []
    joint_list = []

    for i in trange(len(idx_list) - 1):
        start_idx, end_idx = idx_list[i], idx_list[i + 1]
        repeat_times = end_idx - start_idx
        cfg.dest_normals = dest_normals.unsqueeze(0).repeat(repeat_times, 1, 1).to(device).to(cfg.dtype)
        cfg.dest_points = dest_points.unsqueeze(0).repeat(repeat_times, 1, 1).to(device).to(cfg.dtype)
        sample_points = torch.from_numpy(points[start_idx: end_idx]).to(device).to(cfg.dtype)
        sample_normals = torch.from_numpy(normals[start_idx: end_idx]).to(device).to(cfg.dtype)

        ## Generate initial pose and copy source point cloud
        src_points_, src_normals_, sample_T = sample_position(src_points,
                                                              src_normals,
                                                              sample_points,
                                                              sample_normals,
                                                              cfg)
        src_weights_ = src_weights.repeat((repeat_times, 1))

        final_pose, final_joint = dipgrasp(src_points_,
                                           src_normals_,
                                           src_weights_,
                                           sample_T,
                                           gripper,
                                           cfg)
        collision_point_num = grasp_collision_check_avoid_collision(final_pose, final_joint, gripper, cfg)
        collision_free_idx = collision_point_num < cfg.algo_params.collision_filter_threshold
        final_pose = final_pose[collision_free_idx]
        final_joint = final_joint[collision_free_idx]
        if final_pose.shape[0] == 0: continue

        if len(pose_list) == 0:
            pose_list = np.asarray(final_pose.detach().cpu().numpy())
            joint_list = np.asarray(final_joint.detach().cpu().numpy())
        else:
            pose_list = np.vstack((pose_list, np.asarray(final_pose.detach().cpu().numpy())))
            joint_list = np.vstack((joint_list, np.asarray(final_joint.detach().cpu().numpy())))

        if cfg.visualization_params.visualize:
            src_points_vis, src_normals_vis, _ = gripper.compute_pcd(final_pose,
                                                                     final_joint,
                                                                     cfg.link_complete_points,
                                                                     cfg.link_complete_normals)
            for idx, src_points_vis_ in enumerate(src_points_vis):
                visualize_point_cloud(src_points_vis_.detach().cpu().numpy(), dest_points)
    return pose_list, joint_list


@hydra.main(version_base="v1.2", config_path='conf', config_name='default')
def main(cfg: DictConfig):
    assert osp.exists(cfg.datafile), 'The input file is not exist'
    if not osp.exists(cfg.savepath):
        os.makedirs(cfg.savepath)

    device = cfg.device
    data_file = cfg.datafile
    save_path = cfg.savepath
    gripper_name = cfg.gripper.name
    assert gripper_name in ['barrett', 'svh', 'shadow']

    gripper = Gripper(gripper_name)

    algo_param_dict = DictConfig({**cfg.algo_params, **cfg.gripper.optim_params})
    algo_cfg = AlgoConfig(cfg.visualize_setting, algo_param_dict, device=device)

    algo_cfg.init_joint_param(gripper)
    algo_cfg.init_sample_config(gripper)
    algo_cfg.link_partial_points, algo_cfg.link_partial_normals = gripper.get_link_pcd_from_xml()
    algo_cfg.link_complete_points, algo_cfg.link_complete_normals = gripper.get_link_pcd_from_mesh()
    # sample point
    ptCloud = load_obj(data_file)

    poses, joints = generate_grasp_for_obj(ptCloud, gripper, algo_cfg)
    # convex decomposition
    if cfg.simulator:
        origin_obj = trimesh.load(data_file)
        vhacd_objs = trimesh.decomposition.convex_decomposition(origin_obj, maxConvexHulls=32)

        _, filename = os.path.split(data_file)
        tmp_path = osp.join(save_path, 'tmp')

        os.makedirs(tmp_path, exist_ok=True)
        vhacd_path = osp.join(tmp_path, 'vhacd_' + filename)
        export_meshes(vhacd_objs, vhacd_path)
        poses, joints = rfu_filter(poses, joints, gripper_name, vhacd_path)
        if cfg.visualize_setting.visualize_after_simulator:
            poses_ = torch.from_numpy(poses).to(device).float()
            joints_ = torch.from_numpy(joints).to(device).float()
            src_points_vis, src_normals_vis, _ = gripper.compute_pcd(poses_,
                                                                     joints_,
                                                                     algo_cfg.link_complete_points,
                                                                     algo_cfg.link_complete_normals)
            dest_points = torch.from_numpy(np.array(ptCloud.points))
            for idx, src_points_vis_ in enumerate(src_points_vis):
                visualize_point_cloud(src_points_vis_.detach().cpu().numpy(), dest_points)

    np.save(osp.join(save_path, 'pose.npy'), poses)
    np.save(osp.join(save_path, 'joint_state.npy'), joints)





if __name__ == '__main__':
    main()

