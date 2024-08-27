import torch
import numpy as np

from typing import Tuple

from utils import  visualize_point_cloud, link_points
from core.utils import get_transform_mat, farthest_point_sample_index
from core.config import AlgoConfig
from core.geometry.gripper import Gripper
from core.geometry.matchNearestPoints import match_nearest_points

def loss_function(T: torch.Tensor,
                  q: torch.Tensor,
                  pair_map: torch.Tensor,
                  dest_points: torch.Tensor,
                  dest_normals: torch.Tensor,
                  src_weights: torch.Tensor,
                  gripper: Gripper,
                  cfg: AlgoConfig) -> torch.Tensor:
    device = cfg.device
    batch_size = q.shape[0]

    T_delta = torch.einsum('ijk, ikl-> ijl', get_transform_mat(q[:, :6]), T)
    src_points, src_normals, _ = gripper.compute_pcd(T_delta,
                                         q[:, 6:],
                                         cfg.link_partial_points,
                                         cfg.link_partial_normals)

    src_point_num = src_points.shape[1]

    point_dist = torch.linalg.norm(src_points - dest_points, dim = 2)

    ## compute force closure error
    tmp_idx = torch.arange(0, batch_size).repeat(src_point_num // 5, 1).T

    nearest_points_idx = torch.argsort(point_dist, dim = 1)[:, :src_point_num // 5]
    fc_points = src_points[tmp_idx, nearest_points_idx]
    fc_normals = src_normals[tmp_idx, nearest_points_idx]

    tmp_idx = torch.arange(0, batch_size).repeat(cfg.algo_params.fc_point_num, 1).T
    fc_idx = farthest_point_sample_index(fc_points, cfg.algo_params.fc_point_num)
    fc_points = fc_points[tmp_idx, fc_idx]
    fc_normals = fc_normals[tmp_idx, fc_idx]
    Efc1 = torch.sum(fc_normals, dim = 1)
    Efc2 = torch.sum(torch.cross(fc_points, fc_normals, dim = 2), dim = 1)
    Efc = torch.linalg.norm(Efc1, dim = 1) + torch.linalg.norm(Efc2, dim = 1)

    Ep = (src_weights * torch.sum((src_points - dest_points) * dest_normals, dim = 2)) ** 2
    En = (src_weights * (torch.sum(src_normals * dest_normals, dim = 2) + torch.ones((src_points.shape[:2])).to(device))) ** 2
    mask = pair_map > 0

    Ep = cfg.algo_params.ep_coeff * torch.sum(Ep * mask)
    En = cfg.algo_params.en_coeff * torch.sum(En * mask)
    Efc = cfg.algo_params.efc_coeff * torch.sum(Efc)

    ipc_mask = (point_dist < cfg.algo_params.barrier_threshold) & (pair_map > 0)
    point_dist_ = point_dist * ipc_mask + cfg.algo_params.barrier_threshold * torch.ones_like(point_dist) * (~ipc_mask)
    barrier_term = cfg.algo_params.barrier_coeff * torch.sum( -(point_dist_ - cfg.algo_params.barrier_threshold) ** 2 * torch.log(point_dist_ / cfg.algo_params.barrier_threshold))
    limit_dist = torch.minimum(q[:, 6:] - cfg.joints_limit_lower.unsqueeze(0), cfg.joints_limit_upper.unsqueeze(0) - q[:, 6:])
    limit_threshold = cfg.joints_limit_barrier.repeat((batch_size, 1))
    limit_mask = limit_dist < limit_threshold
    limit_dis_new = limit_dist * limit_mask + limit_threshold * torch.ones_like(limit_dist) * (~limit_mask)
    limit_penalty = cfg.algo_params.joints_limit_coeff * torch.sum(- (limit_dis_new - limit_threshold + 0.05) ** 2 * torch.log((limit_dis_new + 0.05) / limit_threshold), dim = 1)
    limit_penalty = torch.sum(limit_penalty)


    # print(Ep.object, En.object, Efc.object, barrier_term.object, limit_penalty.object)

    loss = (Ep + En + Efc + limit_penalty + barrier_term) / src_point_num
    # print(loss)


    return loss

def dipgrasp(src_points: torch.Tensor,
             src_normals: torch.Tensor,
             src_weight: torch.Tensor,
             sample_T: torch.Tensor,
             gripper: Gripper,
             cfg: AlgoConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    device = cfg.device
    batch_size = src_points.shape[0]
    cfg.tmp_decay = 1

    Tcurrent = sample_T.clone()
    joints_states = cfg.joint_init_value.repeat((batch_size, 1))
    q = torch.zeros(batch_size, 6 + joints_states.shape[1]).to(device)
    q[:, 6:] += joints_states
    q.requires_grad_(True)

    for i in range(cfg.algo_params.loop_1_num):

        # match the corresponding point-pair
        pair_map = match_nearest_points(src_points, src_normals, cfg)

        src_weight_match = torch.zeros_like(src_weight).to(device)
        src_weight_match[pair_map > 0] = src_weight[pair_map > 0]
        dest_normals_match = cfg.dest_normals
        dest_points_match = cfg.dest_points
        dest_tmp_mask = pair_map
        dest_tmp_mask[pair_map < 0] = 0

        view_shape = list(dest_tmp_mask.shape)
        view_shape[1:] = [1] * (len(view_shape) - 1)
        repeat_shape = list(dest_tmp_mask.shape)
        repeat_shape[0] = 1
        batch_indices = torch.arange(batch_size, dtype=torch.long).view(view_shape).repeat(repeat_shape)

        dest_normals_match = dest_normals_match[batch_indices, dest_tmp_mask]
        dest_points_match = dest_points_match[batch_indices, dest_tmp_mask]
        dest_normals_match[pair_map < 0, :] = 0
        dest_points_match[pair_map < 0, :] = 0

        for i in range(cfg.algo_params.loop_2_num):

            if cfg.visualization_params.visualize_detail:
                if cfg.visualize_step_count == 0:
                    SrcPC, SrcN, _ = gripper.compute_pcd(Tcurrent, q[:, 6:], cfg.link_partial_points,
                                                         cfg.link_partial_normals)
                    tmp_world_PC = np.concatenate(
                        (SrcPC[0].detach().cpu().numpy(), cfg.dest_points[0].detach().cpu().numpy()),
                        axis=0)
                    visualize_point_cloud(tmp_world_PC,
                                          link_points(SrcPC[0][pair_map[0] > 0].detach().cpu().numpy(),
                                          (dest_points_match[0][pair_map[0] > 0].detach().cpu()).numpy())
                                          )
                    tmp_idx = torch.arange(0, batch_size).repeat(SrcPC.shape[1] // 3, 1).T

                    # point_dist = torch.linalg.norm(src_points - dest_points_match, dim=2)
                    # nearest_points_idx = torch.argsort(point_dist, dim=1)[:, :SrcPC.shape[1] // 3]
                    # fc_points = src_points[tmp_idx, nearest_points_idx]
                    #
                    # visualize_point_cloud(src_points[0].detach().cpu().numpy(), fc_points[0].detach().cpu().numpy())
                cfg.visualize_step_count += 1
                if cfg.visualize_step_count >= cfg.visualization_params.visualize_every_step:
                    cfg.visualize_step_count = 0

            loss = loss_function(Tcurrent,
                                 q,
                                 pair_map,
                                 dest_points_match,
                                 dest_normals_match,
                                 src_weight_match,
                                 gripper,
                                 cfg)
            loss.backward()
            grad_q = q.grad
            grad_q = torch.clamp(grad_q.nan_to_num(nan = 1e-1), -1e-1, 1e-1)

            # update transformation
            tmp = torch.zeros((batch_size, 6)).to(device)
            tmp[:, 3:6] = - cfg.tmp_decay * cfg.algo_params.update_step_t * grad_q[:, 3:6]  # t
            tmp[:, :3] = - cfg.tmp_decay * cfg.algo_params.update_step_R * grad_q[:, :3]  # R
            deltaT = get_transform_mat(tmp)
            Tcurrent = torch.einsum('ijk, ikl-> ijl', deltaT, Tcurrent)

            # update joint_states
            q = q.detach()
            q[:, 6:] -= cfg.tmp_decay * cfg.algo_params.update_step_q * grad_q[:, 6:]
            q.requires_grad_(True)
            cfg.tmp_decay *= cfg.algo_params.decay_rate

        joints_states = q[:, 6:]
        src_points, src_normals, _ = gripper.compute_pcd(Tcurrent,
                                                         joints_states,
                                                         cfg.link_partial_points,
                                                         cfg.link_partial_normals)

    return Tcurrent.detach(), joints_states.detach()