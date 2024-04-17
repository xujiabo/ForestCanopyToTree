import numpy as np
import open3d as o3d
import torch

from algorithm import uv_2
from utils import to_o3d_pcd, yellow, square_distance, blue


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

    return crown_xyz, max_z


def load_graph(path, max_z):
    data = np.load(path)
    data[0, 3] = -1
    xyz, fa, width, is_leaf, leaf_size = data[:, :3], data[:, 3:4], data[:, 4:5], data[:, 5:6], data[:, 6:7]
    max_z = np.max(xyz, axis=0)[2]
    xyz = xyz / max_z
    width, leaf_size = width / max_z, leaf_size / max_z
    data = np.concatenate([xyz, fa, width, is_leaf, leaf_size], axis=1)
    return data


def get_lines(data):
    pts = data[:, :3]
    lines = []
    for i in range(1, data.shape[0]):
        st, ed = pts[i], pts[int(data[i, 3].item())]
        length = np.linalg.norm(st-ed, axis=0).item()
        cyl = o3d.geometry.TriangleMesh.create_cylinder(0.0025, length, resolution=4)
        cyl.paint_uniform_color(blue())
        P = np.array([
            [0, 0, np.linalg.norm(st-ed).item()/2],
            [0, 0, 0],
            [0, 0, -np.linalg.norm(st-ed).item()/2],
        ])
        Q = np.array([
            st.tolist(),
            ((st+ed)/2).tolist(),
            ed.tolist()
        ])
        p_, q_ = P.mean(axis=0).reshape(1, -1), Q.mean(axis=0).reshape(1, -1)
        p, q = P-p_, Q-q_
        H = p.T.dot(q)
        U, Sigma, V = np.linalg.svd(H, compute_uv=True)
        V = V.T

        R = V.dot(U.T)

        v_neg = np.copy(V)
        v_neg[:, 2] = v_neg[:, 2] * -1
        rot_mat_neg = v_neg @ U.T

        R = R if np.linalg.det(R) > 0 else rot_mat_neg

        cyl.rotate(R)
        cyl.translate((st+ed)/2)

        lines.append(cyl)

    return lines


if __name__ == '__main__':
    forest_name = "forest6"
    item = 76

    crown_path = "./10-forest/%s/crown/%d.npy" % (forest_name, item)
    graph_path = "./Ours_Result/%s/graph/%d.npy" % (forest_name, item)

    crown_xyz, max_z = load_crown(crown_path)

    data = load_graph(graph_path, max_z)
    xyz = data[:, :3]

    # 和proposed的consistency
    consistency = square_distance(torch.Tensor(crown_xyz).unsqueeze(0), torch.Tensor(xyz).unsqueeze(0))[0].min(dim=1)[0].numpy()
    print("consistency: %.9f" % np.mean(consistency) * 1000)

    bark_wo_uv, leaf_wo_uv, bark, leaf = uv_2.to_mesh(data)
    bark_wo_uv.compute_vertex_normals()

    crown_pcd = []
    for i in range(crown_xyz.shape[0]):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005, resolution=20)
        sphere.vertex_colors = o3d.utility.Vector3dVector([yellow() for _ in range(len(sphere.vertices))])
        np.asarray(sphere.vertices)[:] += crown_xyz[i]
        sphere.compute_vertex_normals()
        crown_pcd.append(sphere)

    # 构造我方节点图
    graph_nodes = []
    for i in range(data.shape[0]):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.005, resolution=5)
        if data[i, 5] == 1:
            sphere.vertex_colors = o3d.utility.Vector3dVector([[0, 1, 0] for _ in range(len(sphere.vertices))])
        np.asarray(sphere.vertices)[:] += xyz[i]
        sphere.compute_vertex_normals()
        sphere.compute_triangle_normals()
        graph_nodes.append(sphere)
    lines = get_lines(data)

    # o3d.visualization.draw_geometries(graph_nodes, window_name="graph nodes", width=1000, height=1000)

    o3d.visualization.draw_geometries(crown_pcd + [bark_wo_uv], window_name="crown and trunk mesh", width=1000, height=1000)
    o3d.visualization.draw_geometries([bark_wo_uv], window_name="trunk mesh", width=1000, height=1000)
    o3d.visualization.draw_geometries([bark, leaf], window_name="tree", width=1000, height=1000)

    o3d.visualization.draw_geometries(crown_pcd, window_name="crown", width=1000, height=1000)

    o3d.visualization.draw_geometries(graph_nodes+lines, window_name="branch graph", width=1000, height=1000)

