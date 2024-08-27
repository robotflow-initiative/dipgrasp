import numpy as np
import torch
import urdfpy
from typing import Dict

from core.config import AlgoConfig
from core.geometry.gripper import Gripper

def grasp_collision_check_avoid_collision(baseT: torch.Tensor,
                                          jointPara: torch.Tensor,
                                          gripper: Gripper,
                                          cfg: AlgoConfig,
                                          scale: float = 1.0) -> torch.Tensor:
    Ts = gripper.compute_relative_matrix(baseT, jointPara)

    bs = cfg.dest_points.shape[0]
    destPoints = cfg.dest_points
    device = destPoints.device
    bbox = gripper.bbox

    t_w = torch.dstack((destPoints, torch.ones((baseT.shape[0], destPoints.shape[1], 1)).to(device)))
    points_sum = torch.zeros((bs,)).to(device)
    for link in gripper.links:
        if len(cfg.link_complete_points[link]) <= 0: continue
        pose = Ts[link]
        pose_inv = torch.linalg.inv(pose)
        t_g = torch.einsum('ijk,ilk->ijl', t_w, pose_inv)
        t_g = torch.einsum('ijk,kl->ijl', t_g, torch.Tensor(bbox[link][0].T).to(device))
        inBox, collide_object_idx = check_box(t_g, bbox[link][1], scale)
        points_delta = torch.sum(collide_object_idx, dim = 1)
        points_sum += points_delta
    points_sum += check_link_collision(Ts, baseT, gripper, cfg, scale)
    return points_sum

def check_link_collision(Ts: Dict[urdfpy.Link, torch.Tensor],
                         baseT: torch.Tensor,
                         gripper: Gripper,
                         cfg: AlgoConfig,
                         scale: float = 1.0) -> torch.Tensor:
    bs = baseT.shape[0]
    device = baseT.device
    links = gripper.links
    bbox = gripper.bbox
    points_sum = torch.zeros((bs,)).to(device)

    for link1 in links:
        if len(cfg.link_complete_points[link1]) <= 0: continue
        pose_inv = torch.linalg.inv(Ts[link1])
        for link2 in links:
            if link1 == link2 or link1 == gripper.link_parent[link2] or link2 == gripper.link_parent[link1]:
                continue
            link2Points = torch.from_numpy(cfg.link_complete_points[link2]).to(device).to(torch.float)
            link2Points = torch.hstack((link2Points, torch.ones((link2Points.shape[0], 1)).to(device))).repeat((bs, 1, 1))
            tmpPoints = torch.einsum('ijk,ilk->ijl', link2Points, pose_inv)
            tmpPoints = torch.einsum('ijk,kl->ijl', tmpPoints, torch.Tensor(bbox[link1][0].T).to(device))
            inBox, collide_object_idx = check_box(tmpPoints, bbox[link1][1], scale)
            points_sum += torch.sum(collide_object_idx, dim = 1)

    return points_sum

def check_box(points: torch.Tensor,
              box2: np.ndarray,
              scale: float = 1.0):
    device = points.device
    box = 0.5 * scale * torch.from_numpy(box2).to(device)
    inbox = (points[:, :, 0] >= -box[0]) & (points[:, :, 0] <= box[0])\
            & (points[:, :, 1] >= -box[1]) & (points[:, :, 1] <= box[1])\
            & (points[:, :, 2] >= -box[2]) & (points[:, :, 2] <= box[2])
    return torch.any(inbox.long(), dim=1), inbox
