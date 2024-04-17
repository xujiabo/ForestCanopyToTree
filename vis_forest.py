import numpy as np
import open3d as o3d
import torch

from algorithm import uv_2, kruskal
from utils import to_o3d_pcd, yellow, square_distance, blue


def renorm(xyz, center, h):
    xyz[:, 2] += 1
    xyz = xyz / 2
    xyz = xyz * h
    xyz = xyz + center
    return xyz

def load_crown(path):
    crown_xyz = np.load(path)
    crown_xyz, h = crown_xyz[:-1, :], crown_xyz[-1, 0].item()
    # normalize crown
    max_ = crown_xyz.max(axis=0)
    max_z = max_[2]
    crown_xyz_c = crown_xyz.mean(axis=0)
    floor_h = max_z - h
    crown_xyz_c = np.array([crown_xyz_c[0].item(), crown_xyz_c[1].item(), floor_h])
    crown_xyz = crown_xyz - crown_xyz_c.reshape(1, 3)
    crown_xyz = crown_xyz / h

    crown_xyz = crown_xyz * 2
    crown_xyz[:, 2] -= 1

    return crown_xyz, h, crown_xyz_c.reshape(1, 3)


def load_graph(path):
    data = np.load(path)
    xyz, fa, width, is_leaf, leaf_size = data[:, :3], data[:, 3:4], data[:, 4:5], data[:, 5:6], data[:, 6:7]
    max_z = np.max(xyz, axis=0)[2]
    xyz = xyz / max_z
    width, leaf_size = width / max_z, leaf_size / max_z

    xyz = xyz * 2
    xyz[:, 2] -= 1
    # predicted by GNN
    f = torch.zeros((4096 * 4096,))
    f[torch.arange(4096) * 4096 + torch.LongTensor(fa).view(-1)] = 1
    f = f.view(4096, 4096)
    # All attribute are predicted by Networks
    return xyz, f, width, is_leaf, leaf_size


if __name__ == '__main__':
    forest_id = 1

    forest_name = "forest%d" % forest_id
    crown_nums = [357, 240, 489, 218, 310, 328, 311, 149, 495, 319][forest_id-1]
    dem = o3d.io.read_triangle_mesh("./10-forest/forest%d/dem.ply" % forest_id)
    dem.compute_vertex_normals()

    trees = []
    crowns = []

    for item in range(crown_nums):

        crown_path = "./10-forest/%s/crown/%d.npy" % (forest_name, item+1)
        graph_path = "./Ours_Result/%s/graph/%d.npy" % (forest_name, item+1)

        crown_xyz, h, crown_xyz_c = load_crown(crown_path)

        xyz, f, width, is_leaf, leaf_size = load_graph(graph_path)

        width_pred, leaf_size_pred = width * h / 2, leaf_size * h / 2
        crown_xyz, recon_xyz = renorm(crown_xyz, crown_xyz_c, h), renorm(xyz, crown_xyz_c, h)

        # n x n
        dis_gnn = -torch.softmax(f, dim=-1)
        fa_gnn = kruskal.mst(torch.Tensor(recon_xyz).cuda(), dis_gnn.cuda(), k=3)

        data = np.concatenate([
            recon_xyz, fa_gnn.reshape(-1, 1).cpu().numpy(), width_pred,
            is_leaf, leaf_size_pred
        ], axis=1)
        _, _, bark, leaf = uv_2.to_mesh(data)

        trees.append(bark)
        trees.append(leaf)

        crown_color = np.random.rand(3)
        crown_pcd = to_o3d_pcd(crown_xyz, crown_color)

        crowns.append(crown_pcd)

        print("\r%d / %d" % (item+1, crown_nums))
    print()

    o3d.visualization.draw_geometries(crowns+trees+[dem], window_name="crowns and trees", width=1000, height=800)
    o3d.visualization.draw_geometries(trees+[dem], window_name="trees", width=1000, height=800)

