import torch
import numpy as np
import open3d as o3d

def get_point_cloud(points: np.ndarray, normals: np.ndarray) -> o3d.geometry.PointCloud:
    ptCloud = o3d.geometry.PointCloud(o3d.cpu.pybind.utility.Vector3dVector(points))
    ptCloud.normals = o3d.cpu.pybind.utility.Vector3dVector(normals)
    return ptCloud

def construct_4x4_matrix(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    device = R.device
    matrix = torch.zeros((R.shape[0], 4, 4)).to(device)
    matrix[:, 3, 3] += torch.ones(R.shape[0]).to(device)
    matrix[:, :3, :3] += R
    matrix[:, :3, 3] += t.squeeze(1)
    return matrix

def dot_py(A: torch.Tensor, B: torch.Tensor, axis: int) -> torch.Tensor:
    return torch.sum(A.conj() * B, dim=axis)

def get_duplicates(X: np.ndarray) -> np.ndarray:
    _, I = np.unique(X, return_index=True)
    x = np.arange(X.shape[0])
    x = np.delete(x, I)
    duplicates = np.unique(X[x])
    return duplicates

def farthest_point_sample_index(points: torch.Tensor, npoint: int) -> torch.Tensor:
    """
    Args:
        points: point cloud with shape (B, N, 3)
        npoint: number of sampled points

    Returns:
        centroids: sampled point cloud index with shape (B, npoint)
    """
    device = points.device
    B, N, D = points.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = points[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((points - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids
def get_transform_mat(par: torch.Tensor) -> torch.Tensor:
    device = par.device
    batch_size = par.shape[0]
    batch_ones = torch.ones(batch_size).to(device)
    r = par[:, 0:3]
    t = par[:, 3:6]
    c1 = torch.cos(r[:, 0])
    s1 = torch.sin(r[:, 0])
    c2 = torch.cos(r[:, 1])
    s2 = torch.sin(r[:, 1])
    c3 = torch.cos(r[:, 2])
    s3 = torch.sin(r[:, 2])
    Rx = torch.zeros((batch_size, 3, 3)).to(device)
    Rx[:, 0, 0] += batch_ones
    Rx[:, 1, 1] += c1
    Rx[:, 1, 2] += -s1
    Rx[:, 2, 1] += s1
    Rx[:, 2, 2] += c1

    Ry = torch.zeros((batch_size, 3, 3)).to(device)
    Ry[:, 1, 1] += batch_ones
    Ry[:, 0, 0] += c2
    Ry[:, 0, 2] += s2
    Ry[:, 2, 0] += -s2
    Ry[:, 2, 2] += c2

    Rz = torch.zeros((batch_size, 3, 3)).to(device)
    Rz[:, 2, 2] += batch_ones
    Rz[:, 0, 0] += c3
    Rz[:, 0, 1] += -s3
    Rz[:, 1, 0] += s3
    Rz[:, 1, 1] += c3

    R = torch.einsum('ijk,ikl->ijl',torch.einsum('ijk,ikl->ijl', Rx,  Ry), Rz)
    M = construct_4x4_matrix(R, t)
    return M