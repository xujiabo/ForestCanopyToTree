import numpy as np
import open3d as o3d
import torch
import pickle
from scipy.spatial.transform import Rotation


def yellow():
    return [1, 0.706, 0]


def blue():
    return [0, 0.651, 0.929]


def load_obj(path):
    """
    read a dictionary from a pickle file
    """
    with open(path, 'rb') as f:
        return pickle.load(f)


def processbar(current, totle):
    process_str = ""
    for i in range(int(20 * current / totle)):
        process_str += "█"
    while len(process_str) < 20:
        process_str += " "
    return "%s|   %d / %d" % (process_str, current, totle)


def rot2q(rot):
    # Tensor 3 x 3
    r = Rotation.from_dcm(rot)
    return r.as_quat()


def q2rot(q):
    r = Rotation.from_quat(q)
    return r.as_dcm()

def R_from_two_direction(origin_vector, location_vector):
    c = np.dot(origin_vector, location_vector)
    n_vector = np.cross(origin_vector, location_vector)
    s = np.linalg.norm(n_vector)

    n_vector_invert = np.array((
        [0, -n_vector[2], n_vector[1]],
        [n_vector[2], 0, -n_vector[0]],
        [-n_vector[1], n_vector[0], 0]
    ))
    I = np.eye(3)
    # 核心公式：见上图
    R_w2c = I + n_vector_invert + np.dot(n_vector_invert, n_vector_invert) / (1 + c)
    return R_w2c


def to_array(tensor):
    """
    Conver tensor to array
    """
    if not isinstance(tensor, np.ndarray):
        if tensor.device == torch.device('cpu'):
            return tensor.numpy()
        else:
            return tensor.cpu().numpy()
    else:
        return tensor


def to_o3d_pcd(xyz, colors=None):
    """
    Convert tensor/array to open3d PointCloud
    xyz:       [N, 3]
    """
    pcd = o3d.geometry.PointCloud()
    pts = to_array(xyz)
    pcd.points = o3d.utility.Vector3dVector(pts)
    if colors is not None:
        pcd.colors = o3d.utility.Vector3dVector(np.array([colors]*pts.shape[0]))
    return pcd


def to_o3d_feats(embedding):
    """
    Convert tensor/array to open3d features
    embedding:  [N, 3]
    """
    feats = o3d.registration.Feature()
    feats.data = to_array(embedding).T
    return feats


def to_tensor(array):
    """
    Convert array to tensor
    """
    if not isinstance(array, torch.Tensor):
        return torch.from_numpy(array).float()
    else:
        return array


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    # farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    farthest = torch.zeros((B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
        # print("\rfps process: %d / %d" % (i+1, npoint), end="")
    # print()
    return centroids