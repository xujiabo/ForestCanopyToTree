import numpy as np
import open3d as o3d
import cv2
from utils import R_from_two_direction
from utils import to_o3d_pcd
from procedrual_tree_modeling.lpy import Rotation_Matrix_from_x_rotation, Rotation_Matrix_from_y_rotation, compute_angels_from_dir, compute_Rotation_Matrix, Rotation_Matrix_from_z_rotation
# 77画图


def to_mesh(data):
    xyz, fa, width, is_leaf, leaf_size = data[:, :3], data[:, 3], data[:, 4], data[:, 5], data[:, 6]
    fa = fa.astype(np.int32)
    tree = [[] for _ in range(xyz.shape[0])]
    for i in range(0, xyz.shape[0]):
        fa_id = fa[i]
        if fa_id != -1:
            tree[fa_id].append(i)
    bark_texture = cv2.imread("algorithm/bark.jpg")
    bark_texture = cv2.cvtColor(bark_texture, cv2.COLOR_BGR2RGB)
    leaf_texture = cv2.imread("algorithm/leaf.png", cv2.IMREAD_UNCHANGED)
    leaf_texture = cv2.cvtColor(leaf_texture, cv2.COLOR_BGRA2RGBA)

    unit_ring = np.array([
        [1, 0, 0],
        [np.sqrt(2)/2, np.sqrt(2)/2, 0],
        [0, 1, 0],
        [-np.sqrt(2)/2, np.sqrt(2)/2, 0],
        [-1, 0, 0],
        [-np.sqrt(2)/2, -np.sqrt(2)/2, 0],
        [0, -1, 0],
        [np.sqrt(2)/2, -np.sqrt(2)/2, 0],
    ])

    vec = np.zeros((xyz.shape[0], 8, 3))
    vec[0] = unit_ring * width[0]
    vec[0] += xyz[0]
    cur_dir = np.array([0, 0, 1])

    is_leaf = np.round(is_leaf)
    leaf_num = np.nonzero(is_leaf)[0].reshape(-1).shape[0]
    leaf_vec = np.zeros((leaf_num, 4, 3))
    id_to_leaf_id = np.zeros((xyz.shape[0], ))
    cur_leaf_id = 0
    for i in range(1, xyz.shape[0]):
        dir = (xyz[i] - xyz[fa[i]]) / np.linalg.norm(xyz[i] - xyz[fa[i]]).item()
        R = R_from_two_direction(cur_dir, dir)
        ring = unit_ring * width[i]
        ring = R.dot(ring.T).T
        ring = ring + xyz[i]
        vec[i] = ring

        if is_leaf[i] == 1:
            id_to_leaf_id[i] = cur_leaf_id
            cur_leaf_id += 1


    # vec = vec.reshape(-1, 3)

    tris = []
    leaf_tris = []
    uvs = []
    leaf_uvs = []

    print("dfs...")
    dfs(-1, 0, 0, xyz, is_leaf, leaf_size, vec, leaf_vec, id_to_leaf_id, tris, leaf_tris, uvs, leaf_uvs, tree)
    print("finish")
    vec = vec.reshape(-1, 3)
    tris = np.array(tris)

    leaf_vec = leaf_vec.reshape(-1, 3)
    leaf_tris = np.array(leaf_tris)
    # bark mesh
    bark = o3d.geometry.TriangleMesh()
    bark.vertices = o3d.utility.Vector3dVector(vec)
    bark.triangles = o3d.utility.Vector3iVector(tris)
    # box.compute_vertex_normals()

    uvs = np.array(uvs)
    bark.triangle_uvs = o3d.utility.Vector2dVector(uvs)
    bark.triangle_material_ids = o3d.utility.IntVector([0] * tris.shape[0])
    bark.textures = [o3d.geometry.Image(bark_texture)]
    # leaf mesh
    leaf = o3d.geometry.TriangleMesh()
    leaf.vertices = o3d.utility.Vector3dVector(leaf_vec)
    leaf.triangles = o3d.utility.Vector3iVector(leaf_tris)
    # box.compute_vertex_normals()

    leaf_uvs = np.array(leaf_uvs)
    leaf.triangle_uvs = o3d.utility.Vector2dVector(leaf_uvs)
    leaf.triangle_material_ids = o3d.utility.IntVector([0] * leaf_tris.shape[0])
    leaf.textures = [o3d.geometry.Image(leaf_texture)]
    # leaf.textures = [o3d.io.read_image("leaf.png")]

    # o3d.visualization.draw_geometries([leaf], width=1000, height=800, window_name="tree")
    # o3d.visualization.draw_geometries([bark, leaf], width=1000, height=800, window_name="tree")

    return bark, leaf


def dfs(fa_id, cur_id, level, xyz, is_leaf, leaf_size, vec, leaf_vec, id_to_leaf_id, tris, leaf_tris, uvs, leaf_uvs, tree):
    dx, dy = 1/8, 1
    y_divid = 2
    dy = 1/y_divid
    if fa_id != -1:
        fa_ring = vec[fa_id]
        cur_ring = vec[cur_id]
        for j in range(8):
            # v1 = cur_ring[j]
            v1 = cur_id * 8 + j
            # v2 = cur_ring[(j + 1) % 8]
            v2 = cur_id * 8 + (j + 1) % 8
            # v3 = fa_ring[j]
            v3 = fa_id * 8 + j
            # v4 = fa_ring[(j + 1) % 8]
            v4 = fa_id * 8 + (j + 1) % 8
            tris.append([v3, v2, v1])
            # uvs.extend([[j*dx, 0], [(j+1)*dx, 1], [j*dx, 1]])
            uvs.extend([[j * dx, (level-1) % (y_divid+1)*dy], [(j + 1) * dx, level%(y_divid+1)*dy], [j * dx, level%(y_divid+1)*dy]])
            tris.append([v2, v3, v4])
            # uvs.extend([[(j+1)*dx, 1], [j*dx, 0], [(j+1)*dx, 0]])
            uvs.extend([[(j + 1) * dx, level%(y_divid+1)*dy], [j * dx, (level-1)%(y_divid+1)*dy], [(j + 1) * dx, (level-1)%(y_divid+1)*dy]])

        if is_leaf[cur_id] == 1:
            direction = xyz[cur_id] - xyz[fa_id]
            direction = direction / (np.linalg.norm(direction, axis=0).item() + 1e-8)
            this_dir_z = direction
            this_dir_x = np.cross(direction, np.array([0, 0, 1]))
            this_dir_y = np.cross(this_dir_z, this_dir_x)
            this_Rotation_Matrix = np.array([this_dir_x, this_dir_y, this_dir_z]).transpose()
            this_Rotation_Matrix = (this_Rotation_Matrix).dot(Rotation_Matrix_from_z_rotation(1 * 180))
            this_Rotation_Matrix = this_Rotation_Matrix.dot(Rotation_Matrix_from_x_rotation(np.random.normal(0, 20)))
            this_Rotation_Matrix = this_Rotation_Matrix.dot(Rotation_Matrix_from_y_rotation(np.random.normal(0, 20)))
            dir_x = this_Rotation_Matrix[:, 2] if abs(direction[2]) < 0.707 else this_Rotation_Matrix[:, 1]
            dir_z = this_Rotation_Matrix[:, 0]
            z_roll, z_branch = compute_angels_from_dir(dir_z)
            z_Rotation_Matrix = compute_Rotation_Matrix(z_roll, max(z_branch - np.random.normal(45, 5), -90))
            dir_z = z_Rotation_Matrix[:, 2]

            leaf_pos = xyz[cur_id]
            leaf_siz = leaf_size[cur_id]
            # 2   1
            # 3   0
            leaf_v1 = leaf_pos - dir_x * leaf_siz / 2
            leaf_v2 = leaf_pos - dir_x * leaf_siz / 2 + dir_z * leaf_siz
            leaf_v3 = leaf_pos + dir_x * leaf_siz / 2 + dir_z * leaf_siz
            leaf_v4 = leaf_pos + dir_x * leaf_siz / 2
            # print(leaf_v1.shape)

            leaf_id = int(id_to_leaf_id[cur_id].item())
            leaf_vec[leaf_id, 0] = leaf_v1
            leaf_vec[leaf_id, 1] = leaf_v2
            leaf_vec[leaf_id, 2] = leaf_v3
            leaf_vec[leaf_id, 3] = leaf_v4

            v1 = leaf_id * 4 + 0
            v2 = leaf_id * 4 + 1
            v3 = leaf_id * 4 + 2
            v4 = leaf_id * 4 + 3

            leaf_tris.append([v1, v2, v3])
            leaf_uvs.extend([[0, 1], [0, 0], [1, 0]])
            leaf_tris.append([v1, v3, v4])
            leaf_uvs.extend([[0, 1], [1, 0], [1, 1]])

    if len(tree[cur_id]) != 0:
        for ch_id in tree[cur_id]:
            dfs(cur_id, ch_id, level+1, xyz, is_leaf, leaf_size, vec, leaf_vec, id_to_leaf_id, tris, leaf_tris, uvs, leaf_uvs, tree)
    else:
        pass


if __name__ == '__main__':
    pass