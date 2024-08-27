import torch
import numpy as np

from core.config import AlgoConfig
from core.utils import construct_4x4_matrix

def sample_position_for_thin(SrcPC, SrcN, SampleP, SampleN, SampleVec, surface_dis = 0.15):
    device = SrcPC.device
    n = SampleP.shape[0]
    z_axis = torch.Tensor([0, 0, 1])
    z_axis = torch.tile(z_axis.unsqueeze(0), (n, 1)).to(device)
    SrcPC = torch.tile(SrcPC.unsqueeze(0), (n, 1, 1))
    SrcN = torch.tile(SrcN.unsqueeze(0), (n, 1, 1))
    mean_SrcPC = torch.mean(SrcPC, dim=1)
    pos = SampleP
    SrcPC = SrcPC - torch.tile(mean_SrcPC.unsqueeze(1), (1, SrcPC.shape[1], 1)) + torch.tile(pos.unsqueeze(1),
                                                                                             (1, SrcPC.shape[1], 1))
    mean_SrcPC2 = torch.tile(torch.mean(SrcPC, dim=1).unsqueeze(1), (1, SrcPC.shape[1], 1))
    rz = torch.zeros((n, 3), dtype=torch.double).to(device) - SampleN
    rx = torch.zeros((n, 3), dtype=torch.double).to(device) - SampleVec
    rz -= torch.sum(torch.mul(rx, rz), dim = 1).unsqueeze(1) * rx
    rz /= torch.norm(rz, dim=1).unsqueeze(1)

    module = torch.linalg.norm(rx, axis=1)
    zero_mask = module < 1e-10
    rx[zero_mask, :] = torch.Tensor([-1, 0, 0]).type(dtype=torch.double).to(device)
    rx = rx / torch.tile(torch.linalg.norm(rx, axis=1).unsqueeze(1), (1, 3))
    ry = torch.cross(rz, rx)
    R = torch.cat((rx.unsqueeze(2), ry.unsqueeze(2), rz.unsqueeze(2)), dim=2)
    tmp = torch.einsum('ijk,ikl->ijl', (SrcPC - mean_SrcPC2), torch.permute(R, (0, 2, 1)))
    SrcPC = tmp + mean_SrcPC2 + torch.tile(SampleN.unsqueeze(1), (1, SrcPC.shape[1], 1)) * surface_dis + torch.tile(
        z_axis.unsqueeze(1), (1, SrcPC.shape[1], 1)) * 0.0
    SrcN = torch.einsum('ijk,ikl->ijl', SrcN, torch.permute(R, (0, 2, 1)))
    Tsample = torch.einsum('ijk,ikl->ijl', construct_4x4_matrix(R, pos + SampleN * surface_dis + z_axis * 0.0),
                           construct_4x4_matrix(torch.tile(torch.eye(3).to(device).unsqueeze(0), (n, 1, 1)),
                                                (-mean_SrcPC)))
    return SrcPC, SrcN, Tsample


def sample_position(src_points: torch.Tensor,
                    src_normals: torch.Tensor,
                    sample_points: torch.Tensor,
                    sample_normals: torch.Tensor,
                    cfg: AlgoConfig):
    device = cfg.device

    batch_size = sample_points.shape[0]
    src_points_num = src_points.shape[1]

    src_points_batch = torch.tile(src_points, (batch_size, 1, 1))
    src_normals_batch = torch.tile(src_normals, (batch_size, 1, 1))
    src_points_mean = torch.mean(src_points_batch, dim=1)
    src_points_centered = src_points_batch - torch.tile(src_points_mean.unsqueeze(1), (1, src_points_num, 1))

    t = torch.rand((batch_size, 1)).to(device)
    float_vec = (t * cfg.algo_params.min_distance + (1 - t) * cfg.algo_params.max_distance) * sample_normals

    ## Construct rotation matrix which make the palm opposite to normals
    # Constuct rx vertical to rz
    rz = -sample_normals
    rx = torch.zeros((batch_size, 3), dtype=cfg.dtype).to(device)
    rx[:, 0] = rz[:, 1]
    rx[:, 1] = -rz[:, 0]
    rx_norm = torch.linalg.norm(rx, axis = 1)
    zero_mask = rx_norm < 1e-10
    rx[zero_mask, :] = torch.Tensor([-1, 0, 0]).type(dtype=cfg.dtype).to(device)
    rx = rx / torch.tile(torch.linalg.norm(rx, axis = 1).unsqueeze(1), (1, 3))
    ry = torch.cross(rz, rx, dim = 1)
    R = torch.stack((rx, ry, rz), dim = 2)

    # Add rotation noise
    alpha = torch.rand(batch_size) * 2 * np.pi
    cos_alpha = torch.cos(alpha).to(device)
    sin_alpha = torch.sin(alpha).to(device)
    Rxy = torch.zeros((batch_size, 3, 3)).to(device)
    Rxy[:, 0, 0] = cos_alpha
    Rxy[:, 0, 1] = -sin_alpha
    Rxy[:, 1, 0] = sin_alpha
    Rxy[:, 1, 1] = cos_alpha
    Rxy[:, 2, 2] = torch.ones(batch_size)
    Rxy = Rxy.type(dtype = cfg.dtype)
    R = torch.einsum('ijk,ikl->ijl', R, Rxy)

    ## Make sure that the palm point in the positive direction of the z-axis
    R = correct_coordinate(R, cfg)

    src_points_centered = torch.einsum('ijk,ikl->ijl',
                                       src_points_centered,
                                       torch.permute(R, (0, 2, 1)))
    src_points_batch = src_points_centered + torch.tile(sample_points.unsqueeze(1), (1, src_points_num, 1)) + torch.tile(float_vec.unsqueeze(1), (1, src_points_num, 1))
    src_normals_batch = torch.einsum('ijk,ikl->ijl',
                                     src_normals_batch,
                                     torch.permute(R, (0, 2, 1)))
    Tsample = torch.einsum('ijk,ikl->ijl',
                           construct_4x4_matrix(R,  sample_points + float_vec),
                           construct_4x4_matrix(torch.eye(3).repeat((batch_size, 1, 1)).to(device), (-src_points_mean)))
    return src_points_batch, src_normals_batch, Tsample

def correct_coordinate(rotation_matrix: torch.Tensor, cfg: AlgoConfig) -> torch.Tensor:
    R = rotation_matrix
    device = cfg.device
    batch_size = rotation_matrix.shape[0]

    if 'use_y2z' in cfg.sample_config:
        R_y2z = torch.Tensor([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ]).type(dtype=cfg.dtype).to(device)
        R_y2z = R_y2z.repeat((batch_size, 1, 1))
        R = torch.einsum('ijk,ikl->ijl', R, R_y2z)
    if 'use_y2-z' in cfg.sample_config:
        R_y2_z = torch.Tensor([
            [1, 0, 0],
            [0, 0, 1],
            [0, -1, 0]
        ]).type(dtype=cfg.dtype).to(device)
        R_y2_z = R_y2_z.repeat((batch_size, 1, 1))
        R = torch.einsum('ijk,ikl->ijl', R, R_y2_z)
    if 'use_x2z' in cfg.sample_config:
        R_x2z = torch.Tensor([
            [0, 0, -1],
            [0, 1, 0],
            [1, 0, 0]
        ]).type(dtype=cfg.dtype).to(device)
        R_x2z = R_x2z.repeat((batch_size, 1, 1))
        R = torch.einsum('ijk,ikl->ijl', R, R_x2z)
    if 'use_negz' in cfg.sample_config:
        R_negz = torch.Tensor([
            [-1, 0, 0],
            [0, 1, 0],
            [0, 0, -1]
        ]).type(dtype=cfg.dtype).to(device)
        R_negz = R_negz.repeat((batch_size, 1, 1))
        R = torch.einsum('ijk,ikl->ijl', R, R_negz)

    return R
