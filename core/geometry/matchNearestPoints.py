import torch
from core.utils import dot_py
from core.config import AlgoConfig

inf = 1e10

def match_nearest_points(src_points: torch.Tensor,
                         src_normals: torch.Tensor,
                         cfg: AlgoConfig) -> torch.Tensor:
    ## Calculate closest point for a group of source points.

    dest_points = cfg.dest_points
    dest_normals = cfg.dest_normals

    d_1 = torch.tile(torch.sum(src_points ** 2, dim = 2).unsqueeze(2), (1, 1, dest_points.shape[1]))
    d_2 = torch.tile(torch.sum(dest_points ** 2, dim = 2).unsqueeze(1), (1, src_points.shape[1], 1))
    d_3 = torch.einsum('ijk,ilk->ijl', src_points, dest_points)

    d = d_1 + d_2 - 2 * d_3
    idx = torch.argmin(d, dim = 2)
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(src_points.shape[0], dtype=torch.long).view(view_shape).repeat(repeat_shape)
    match_normals = dest_normals[batch_indices, idx, :]

    if cfg.algo_params.norm_filter:
        idx_qualify = dot_py(match_normals, src_normals, axis = 2) > 0
        idx_qualify = idx_qualify.type(dtype=torch.bool)
        idx[~idx_qualify] = -1

    return idx