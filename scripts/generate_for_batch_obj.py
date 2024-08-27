import os
import sys
import glob
import hydra
import torch
import trimesh
import numpy as np
import os.path as osp

from copy import copy
from loguru import logger
from omegaconf import DictConfig

sys.path.append('.')
from main import generate_grasp_for_obj
from utils import load_obj, export_meshes, rfu_filter, visualize_point_cloud
from core.config import AlgoConfig
from core.geometry.gripper import Gripper

torch.set_default_dtype(torch.float)
gripper_name = 'shadow'

assert gripper_name in ['barrett', 'svh', 'svh_r', 'shadow']


@hydra.main(version_base="v1.2", config_path='../conf', config_name='batch')
def main(cfg: DictConfig):
    logger.add('results.log')
    assert osp.exists(cfg.datapath), 'The input path is not exist'
    assert osp.isdir(cfg.datapath), 'The input path is not a directory'
    if not osp.exists(cfg.savepath):
        os.makedirs(cfg.savepath)

    device = cfg.device
    data_path = cfg.datapath
    save_path = cfg.savepath
    gripper_name = cfg.gripper.name
    assert gripper_name in ['barrett', 'svh', 'svh_r', 'shadow']

    obj_str = '**/aligned_mesh.obj'
    if osp.split(data_path)[1] == 'drink':
        obj_str = '**/drink*.obj'
    elif osp.split(data_path)[1] == 'bowl':
        obj_str = '**/bowl*.obj'
    elif osp.split(data_path)[1] == 'tableware':
        obj_str = '**/tableware*.obj'
    obj_paths = glob.glob(osp.join(data_path, obj_str), recursive=True)

    gripper = Gripper(gripper_name)

    algo_param_dict = DictConfig({**cfg.algo_params, **cfg.gripper.optim_params})
    algo_cfg = AlgoConfig(cfg.visualize_setting, algo_param_dict, device=device)

    algo_cfg.init_joint_param(gripper)
    algo_cfg.init_sample_config(gripper)
    algo_cfg.link_partial_points, algo_cfg.link_partial_normals = gripper.get_link_pcd_from_xml()
    algo_cfg.link_complete_points, algo_cfg.link_complete_normals = gripper.get_link_pcd_from_mesh()
    # sample point

    for idx, obj_path in enumerate(obj_paths):
        print(f'Processing {idx}th object')
        save_dirs = obj_path.split(os.sep)
        tmp_path = copy(save_path)
        for i in range(-3, -1):
            tmp_path = osp.join(tmp_path, save_dirs[i])
            if not osp.exists(tmp_path):
                os.mkdir(tmp_path)

        obj_name = osp.split(osp.split(obj_path)[0])[1]
        logger.info(f'Processing obj: {obj_name}')
        # sample point
        ptCloud = load_obj(obj_path)
        poses, joints = generate_grasp_for_obj(ptCloud, gripper, algo_cfg)
        assert poses.shape[0] == joints.shape[0]
        logger.info(f'Generate {poses.shape[0]} valid grasps')
        if cfg.simulator:
            origin_obj = trimesh.load(obj_path)
            vhacd_objs = trimesh.decomposition.convex_decomposition(origin_obj, maxConvexHulls=32)

            _, filename = os.path.split(obj_path)
            tmp_path = osp.join(save_path, 'tmp')

            os.makedirs(tmp_path, exist_ok=True)
            vhacd_path = osp.join(tmp_path, 'vhacd_' + filename)
            export_meshes(vhacd_objs, vhacd_path)
            # try:
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
            logger.info(f'We get {poses.shape[0]} valid grasps on object {obj_name} after simulator filtering')
            # except:
                # logger.info(f'There\'re some problems on obj: {obj_name}')
        np.save(osp.join(tmp_path, 'pose.npy'), poses)
        np.save(osp.join(tmp_path, 'joint_state.npy'), joints)


if __name__ == '__main__':
    main()
