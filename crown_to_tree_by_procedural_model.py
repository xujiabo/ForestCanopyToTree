import numpy as np
import open3d as o3d
from utils import to_o3d_pcd, blue, yellow
# import CSF
from procedrual_tree_modeling.modelling import Control_Para_Aanlysis, Metropolis_Hastings_optimization, from_para_to_tree_mesh
from procedrual_tree_modeling import lpy
from procedrual_tree_modeling import global_node_pool
from procedrual_tree_modeling.lpy import compute_angels_from_dir, compute_Rotation_Matrix, Rotation_Matrix_from_z_rotation, Rotation_Matrix_from_x_rotation, Rotation_Matrix_from_y_rotation
import torch
from utils import farthest_point_sample
import queue

device = torch.device("cuda:0")


def dfs(tree, node_num, node_pos):
    global level, min_layer, max_layer, min_type, max_type, max_level
    node = tree[node_num]
    node_pos.append(node.pos)
    # print("level: %d" % level)
    max_level = max(max_level, level)
    if node_num == 0:
        level += 1
        dfs(tree, node.left_child, node_pos)
        level -= 1
        return
    if node.left_child is not None:
        level += 1
        dfs(tree, node.left_child, node_pos)
        level -= 1
    if node.right_child is not None:
        level += 1
        dfs(tree, node.right_child, node_pos)
        level -= 1


def remake_id(pool):
    q = queue.Queue()

    class Node:
        def __init__(self, node, ind):
            self.node, self.ind = node, ind

        def __str__(self):
            width = self.node.width
            type = self.node.type # [1, 2, 3]
            layer = self.node.layer # [0, 1, 2, 3, 4, 5, 6]

            has_l_ch = 1 if self.node.left_child is not None else 0
            has_r_ch = 1 if self.node.right_child is not None else 0
            l_ch_ind = self.node.left_child if self.node.left_child is not None else -1
            r_ch_ind = self.node.right_child if self.node.right_child is not None else -1
            last_ind = self.node.last_node if self.node.last_node is not None else -1
            l_length = self.node.left_l
            r_length = self.node.right_l
            l_direct = self.node.left_dir if self.node.left_dir is not None else [0, 0, 0]
            r_direct = self.node.right_dir if self.node.right_dir is not None else [0, 0, 0]

            return "width=%f,type=%d,layer=%d,has_l_ch=%d,has_r_ch=%d,l_ch_ind=%d,r_ch_ind=%d,last_ind=%d,l_length=%f,r_length=%f,l_direct=%f;%f;%f,r_direct=%f;%f;%f" \
                   % (width, type, layer, has_l_ch, has_r_ch, l_ch_ind, r_ch_ind, last_ind, l_length, r_length, l_direct[0], l_direct[1], l_direct[2], r_direct[0], r_direct[1], r_direct[2])

    q.put(Node(pool[0], 0))
    cur_id = 0
    line = ""
    new_tree = []
    while not q.empty():
        head = q.get()
        cur_node, cur_node_id = head.node, head.ind
        if cur_node.left_child is not None:
            cur_id += 1
            l_node = pool[cur_node.left_child]
            q.put(Node(l_node, cur_id))
            # 改指针
            l_node.last_node = cur_node_id
            cur_node.left_child = cur_id
        if cur_node.right_child is not None:
            cur_id += 1
            r_node = pool[cur_node.right_child]
            q.put(Node(r_node, cur_id))
            # 改指针
            r_node.last_node = cur_node_id
            cur_node.right_child = cur_id

        line = line + str(head) + ("\n" if not q.empty() else "")
        new_tree.append(cur_node)

    return new_tree

def norm_crown(xyz, h):
    max_ = xyz.max(axis=0)
    max_z = max_[2]
    floor_h = max_z - h
    # h = max_z - floor_h
    center = xyz.mean(axis=0)
    gpc_trans = xyz + np.array([-center[0], -center[1], -floor_h])
    gpc_trans = gpc_trans * (1 / h)
    return gpc_trans, center, h


ts_para = None
leaf_num_internode = 0
leaf_life_time = 0
leaf_drop_possibility = 0
leaf_num = 0
leaf_size_variance = 0.02
leaf_grow_rate = 1.1
leaf_mean_size = 0.2
leafs = []
leafs_sizes = []
pool = None
def traverse_and_compress(p_id,angle_interval=45,
                          tex_height_real_length=0.5, tex_width_real_length=0.5, ty=0, TRIANGLES=True,compress_flag=True,t=12):
    global leafs, leafs_sizes
    p=pool[p_id]
    if p.id==0:
        traverse_and_compress(p.left_child, ty=ty, tex_width_real_length=tex_width_real_length,
                                   tex_height_real_length=tex_height_real_length, compress_flag=compress_flag,t=t)
        return
    if p.type==6:
        return
    l=pool[p.last_node].left_l if pool[p.last_node].left_child==p.id else pool[p.last_node].right_l
    direction = pool[p.last_node].left_dir if pool[p.last_node].left_child == p.id else pool[p.last_node].right_dir
    p1=pool[p.last_node]
    p2=p
    w1 = p1.width
    w2 = p2.width
    layer1 = p1.layer
    layer2 = p2.layer

    if l > 0:
        leaf_counts = 0
        roll, branch = compute_angels_from_dir(direction)
        Rotation_Matrix = compute_Rotation_Matrix(roll, branch)
        node = p2
        leaf_year_old = max(ts_para.get('t') -1- node.created_age,0)
        basic_width=(ts_para.get('IBL') / ts_para.variables_list[ts_para.BDM]*1.001)
        rotation=360/leaf_num_internode
        if leaf_year_old <= leaf_life_time and p2.width<basic_width:
            leaf_size = leaf_mean_size * pow(leaf_grow_rate, leaf_year_old)
            leaf_size = np.random.normal(leaf_size, leaf_size_variance)*ts_para.get('IBL')
            for nn in range(leaf_num_internode):
                for ln in range(leaf_num):
                    rand = np.random.rand()
                    drop_possibility=1-pow(1-leaf_drop_possibility,leaf_year_old)
                    if rand < drop_possibility:
                        continue
                    this_dir_z=direction
                    this_dir_x=np.cross(direction,np.array([0,0,1]))
                    this_dir_y=np.cross(this_dir_z,this_dir_x)
                    this_Rotation_Matrix=np.array([this_dir_x,this_dir_y,this_dir_z]).transpose()
                    this_Rotation_Matrix=(this_Rotation_Matrix).dot(Rotation_Matrix_from_z_rotation(nn*rotation+ln*360/(leaf_num)))
                    this_Rotation_Matrix=this_Rotation_Matrix.dot(Rotation_Matrix_from_x_rotation(np.random.normal(0,20)))
                    this_Rotation_Matrix = this_Rotation_Matrix.dot(
                        Rotation_Matrix_from_y_rotation(np.random.normal(0, 20)))
                    # dir_x=this_Rotation_Matrix[:,2] if abs(direction[2])<0.707 else this_Rotation_Matrix[:,1]
                    dir_z = this_Rotation_Matrix[:,0]
                    z_roll, z_branch = compute_angels_from_dir(dir_z)
                    z_Rotation_Matrix = compute_Rotation_Matrix(z_roll, max(z_branch-np.random.normal(45, 5),-90))
                    dir_z=z_Rotation_Matrix[:,2]
                    #leaf_pos=p2.pos+dir_z*w2
                    leaf_pos = p1.pos +direction*l*(nn+1)/leaf_num_internode+dir_z * w2
                    leafs.append(leaf_pos)
                    leafs_sizes.append(leaf_size)

    if p.left_child is not None:
        traverse_and_compress(p.left_child, ty=ty, tex_width_real_length=tex_width_real_length,
                                   tex_height_real_length=tex_height_real_length, compress_flag=compress_flag,t=t)
    if p.right_child is not None:
        traverse_and_compress(p.right_child, ty=ty, tex_width_real_length=tex_width_real_length,
                                   tex_height_real_length=tex_height_real_length, compress_flag=compress_flag,t=t)


def generate_leaf(params, ts):
    global ts_para
    global leaf_num_internode
    global leaf_life_time
    global leaf_drop_possibility
    global leaf_num
    global leaf_mean_size
    global pool
    global leafs, leafs_sizes

    leaf_mean_size = 0.75
    leaf_drop_possibility = 0.
    leaf_life_time = 7

    leaf_num = 1
    leaf_life_time = 10
    leaf_num_internode = 2

    ts_para = params
    leafs, leafs_sizes = [], []
    pool = ts.pool

    traverse_and_compress(0, tex_width_real_length=0.5, tex_height_real_length=1.0)

    return leafs, leafs_sizes


def save_graph():
    global level, min_layer, max_layer, min_type, max_type, max_level
    global_pool = []
    for i in range(lpy.MAX_NODES_SIZE * 8):
        '''MAX_NODES_SIZE should be adjusted according to the memory size '''
        node_i = global_node_pool.Mynode()
        node_i.id = i
        global_pool.append(node_i)

    for i in range(0, 357):
        node_num = 0
        while node_num < 4096:
            level = 0
            min_layer, max_layer = 100, -1
            min_type, max_type = 100, -1
            max_level = 0
            for j in range(len(global_pool)):
                global_pool[j].reset_node()

            crown_xyz = np.load("./10-forest/forest1/crown/%d.npy" % i)
            crown_xyz, h = crown_xyz[:-1, :], crown_xyz[-1, 0].item()
            normed_crown_xyz, center, h = norm_crown(crown_xyz, h)

            para_mean, para_std = Control_Para_Aanlysis(normed_crown_xyz, global_pool)
            optimal_para = Metropolis_Hastings_optimization(normed_crown_xyz, para_mean, para_std, "./out", 100, global_pool)
            _, dis, ts = from_para_to_tree_mesh(global_pool, normed_crown_xyz,
                                                para=optimal_para,
                                                this_sample_num=5)

            # pos = []
            # dfs(ts.pool, 0, pos)
            # pos = np.stack(pos, axis=0)

            leafs, leafs_sizes = generate_leaf(optimal_para, ts)
            if len(leafs) == 0:
                print("%d skip..." % i)
                node_num = 0
                continue
            leafs = np.stack(leafs, axis=0)

            tree = remake_id(ts.pool)
            pos = np.zeros((len(tree), 3))
            width = np.zeros((len(tree),))
            fa = np.zeros((len(tree),))
            is_leaf = np.zeros((len(tree),))
            leaf_size = np.zeros((len(tree),))

            for j in range(len(tree)):
                node = tree[j]
                pos[j, :] = node.pos
                width[j] = node.width
                fa[j] = -1 if j == 0 else node.last_node

            fa = fa.astype(np.int)

            xyz_pcd = to_o3d_pcd(pos, yellow())
            pos_kd_tree = o3d.geometry.KDTreeFlann(xyz_pcd)
            for j in range(leafs.shape[0]):
                leaf_pt = leafs[j]
                _, inds, _ = pos_kd_tree.search_knn_vector_3d(leaf_pt, knn=2)
                # ind = inds[0]
                is_leaf[inds] = 1
                leaf_size[inds] = leafs_sizes[j]
            if pos.shape[0] <= 4096:
                print("%d skip..." % i)
                continue
            # 化简
            node_num = 4096 if pos.shape[0] >= 4096 else pos.shape[0]
            print(str(pos.shape) + " -> %d" % node_num)
            ds_inds = farthest_point_sample(torch.Tensor(pos).to(device).unsqueeze(0), node_num)[0].cpu().numpy()
            valid = np.zeros((len(tree),))
            valid[ds_inds] = 1

            if np.nonzero(valid)[0].reshape(-1).shape[0] != node_num:
                print("%d skip..." % i)
                node_num = 0
                continue
            new_id = np.zeros((len(tree),))
            new_id[np.nonzero(valid)[0].reshape(-1)] = np.arange(node_num)

            def find_fa(node_id):
                if node_id == 0:
                    return 0
                fa_ = fa[node_id]
                if valid[fa_] == 1:
                    return fa_
                return find_fa(fa_)

            new_fa = np.zeros((node_num,))
            item = 1
            for j in range(1, len(tree)):
                if valid[j] == 1:
                    fa_id = find_fa(j)
                    new_fa[item] = new_id[fa_id]
                    item += 1
            # new_fa = new_fa.astype(np.int)
            # 重新找爹
            valid_inds = np.nonzero(valid)[0].reshape(-1)
            xyz = pos[valid_inds]
            # print(xyz.shape)
            width = width[valid_inds]
            is_leaf = is_leaf[valid_inds]
            leaf_size = leaf_size[valid_inds]

            data = np.concatenate([
                xyz, new_fa.reshape(-1, 1), width.reshape(-1, 1), is_leaf.reshape(-1, 1), leaf_size.reshape(-1, 1)
            ], axis=1)

            np.save("./procedrual_pred/forest9/graph/%d.npy" % i, data)
            np.save("./procedrual_pred/forest9/params/%d.npy" % i, optimal_para.variables_list)
            print("save finish ! %d / %d" % (i, 495))


if __name__ == '__main__':
    save_graph()