import numpy as np
import open3d as o3d
import torch
from utils import square_distance, to_o3d_pcd, yellow, blue


def find_fa(x, fa):
    # return x if fa[x] == x else find_fa(fa[x], fa)
    root = x
    while fa[root] >= 0:
        root = fa[root]
    while x != root:
        t = fa[x]
        fa[x] = root
        x = t
    return root


def union_fa(root1, root2, fa):
    if root1 == root2:
        return
    if fa[root2] > fa[root1]:
        fa[root1] += fa[root2]
        fa[root2] = root1
    else:
        fa[root2] += fa[root1]
        fa[root1] = root2


def dfs(fa_id, cur_id, tree, fa):
    fa[cur_id] = fa_id
    if len(tree[cur_id]) != 0:
        for i in range(len(tree[cur_id])):
            ch_id = tree[cur_id][i]
            if ch_id != fa_id:
                dfs(cur_id, ch_id, tree, fa)


def mst(pcd, dis, k=10):
    # n x 3, n x n
    # dis = square_distance(pcd.unsqueeze(0), pcd.unsqueeze(0))[0]
    neighbor_dis, neighbor_inds = torch.topk(dis, dim=1, k=k, largest=False)
    # n x k x 1
    neighbor_inds = neighbor_inds[:, :].unsqueeze(2)
    neighbor_dis = neighbor_dis[:, :]
    # n x k x 1
    s = torch.arange(pcd.shape[0]).to(device=pcd.device).unsqueeze(1).repeat([1, k]).unsqueeze(2)
    # (nk, 2)
    edges = torch.cat([s, neighbor_inds], dim=2).view(-1, 2)

    _, sorted_inds = torch.sort(neighbor_dis.contiguous().view(-1), dim=0)
    edges = edges[sorted_inds]

    cur_edge_num = 0
    # fa = torch.arange(pcd.shape[0]).long()
    fa = -torch.ones(pcd.shape[0]).long()
    tree = [[] for _ in range(pcd.shape[0])]
    for i in range(edges.shape[0]):
        s, e = edges[i, 0].item(), edges[i, 1].item()
        fa_s, fa_e = find_fa(s, fa), find_fa(e, fa)
        if fa_s != fa_e:
            # fa[fa_s] = fa_e
            union_fa(fa_s, fa_e, fa)

            tree[s].append(e)
            tree[e].append(s)
            cur_edge_num += 1
            print("\rmit: %d / %d" % (cur_edge_num, pcd.shape[0]-1), end="")

            if cur_edge_num == pcd.shape[0]-1:
                break
    print()

    _, min_z_ind = torch.min(pcd, dim=0)
    root_id = min_z_ind[2].item()
    fa = torch.zeros(pcd.shape[0]).long()
    dfs(-1, root_id, tree, fa)
    return fa.to(pcd.device)


if __name__ == '__main__':
    pass