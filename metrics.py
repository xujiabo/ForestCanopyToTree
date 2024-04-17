import numpy as np
import cv2
import open3d as o3d
from vis_mesh import load_graph, load_crown
from algorithm import uv_2
import torch
from torch.nn import functional as F
import clip
from utils import square_distance


def render(bark, leaf, save_path="."):

    W = 1024

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=W, height=W)

    vis.add_geometry(bark)
    vis.add_geometry(leaf)

    R_y = np.array([
        [np.cos(np.pi / 2), 0, np.sin(np.pi / 2)],
        [0, 1, 0],
        [-np.sin(np.pi / 2), np.cos(np.pi / 2), 0]
    ])
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(-np.pi / 2), -np.sin(-np.pi / 2)],
        [0, np.sin(-np.pi / 2), np.cos(-np.pi / 2)]
    ])
    R_z = np.array([
        [np.cos(np.pi / 2), -np.sin(np.pi / 2), 0],
        [np.sin(np.pi / 2), np.cos(np.pi / 2), 0],
        [0, 0, 1]
    ])

    np.asarray(bark.vertices)[:] = R_x.dot(np.asarray(bark.vertices).T).T
    np.asarray(leaf.vertices)[:] = R_x.dot(np.asarray(leaf.vertices).T).T

    for i in range(4):  # 设置5个视图

        if i > 0:
            # bark.rotate(R_y)
            # leaf.rotate(R_y)
            np.asarray(bark.vertices)[:] = R_y.dot(np.asarray(bark.vertices).T).T
            np.asarray(leaf.vertices)[:] = R_y.dot(np.asarray(leaf.vertices).T).T

        vis.update_geometry(bark)
        vis.update_geometry(leaf)

        vis.poll_events()
        vis.update_renderer()
        # ctr.change_azimuth(i * (th / 4))
        vis.capture_screen_image(save_path+"_v%d.png" % i)
        img = cv2.imread(save_path+"_v%d.png" % i)
        w = 768
        img = img[(W-w)//2:(W-w)//2+w, (W-w)//2:(W-w)//2+w, :]
        cv2.imwrite(save_path+"_v%d.png" % i, img)

    vis.destroy_window()


def get_clip_model(clip_model_type):
    device = torch.device("cuda:0")
    if clip_model_type == "B-16":
        print("Bigger model is being used B-16")
        clip_model, clip_preprocess = clip.load("ViT-B/16", device=device, download_root="./params")
        cond_emb_dim = 512
    elif clip_model_type == "RN50x16":
        print("Using the RN50x16 model")
        clip_model, clip_preprocess = clip.load("RN50x16", device=device, download_root="./params")
        cond_emb_dim = 768
    else:
        clip_model, clip_preprocess = clip.load("ViT-B/32", device=device, download_root="./params")
        cond_emb_dim = 512
    input_resolution = clip_model.visual.input_resolution
    # train_cond_embs_length = clip_model.train_cond_embs_length
    vocab_size = clip_model.vocab_size
    # cond_emb_dim  = clip_model.embed_dim
    # print("Model parameters:", f"{np.sum([int(np.prod(p.shape)) for p in clip_model.parameters()]):,}")
    print("cond_emb_dim:", cond_emb_dim)
    print("Input resolution:", input_resolution)
    # print("train_cond_embs length:", train_cond_embs_length)
    print("Vocab size:", vocab_size)
    return clip_model


def preprocess(images, norm=True):
    mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073]).to(images.device).view(1, 3, 1, 1)
    std = torch.Tensor([0.26862954, 0.26130258, 0.27577711]).to(images.device).view(1, 3, 1, 1)
    if norm:
        images = (images - mean) / std
    images = F.interpolate(images, size=384, mode="bicubic")
    return images


def CLIP_Precision(forest_id, item, clip_model=None, only_v0=True):
    crown_path = "./10-forest/forest%d/crown/%d.npy" % (forest_id, item)
    graph_path = "./Ours_Result/forest%d/graph/%d.npy" % (forest_id, item)

    crown_xyz, max_z = load_crown(crown_path)

    data = load_graph(graph_path, max_z)
    data[:, 2] -= 0.5

    bark_wo_uv, leaf_wo_uv, bark, leaf = uv_2.to_mesh(data)

    render_imgs_save_path = "./tree"
    render(bark, leaf, save_path=render_imgs_save_path)

    if clip_model is None:
        clip_model = get_clip_model("RN50x16")
        clip_model.eval()

    with torch.no_grad():

        prompt = 'a photo of a tree'
        text = clip.tokenize([prompt]).to(torch.device("cuda:0"))
        cond_embedding = clip_model.encode_text(text).detach()
        cond_embedding = cond_embedding / cond_embedding.norm(dim=-1, keepdim=True)
        cond_embedding = cond_embedding.view(-1, 1)

        if only_v0:
            view_list = [0]
        else:
            view_list = [0, 1, 2, 3]
        cp_mean = 0
        for view_id in view_list:
            img = cv2.imread(render_imgs_save_path+"_v%d.png" % view_id)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = torch.Tensor(img.astype(np.float32) / 255).permute([2, 0, 1]).unsqueeze(0).cuda()
            img = preprocess(img)

            imgs_embeddings = clip_model.encode_image(img)
            imgs_embeddings = imgs_embeddings / imgs_embeddings.norm(dim=-1, keepdim=True)

            cp = torch.matmul(imgs_embeddings, cond_embedding).squeeze(dim=1).item()
            cp_mean += cp
        cp_mean = cp_mean / len(view_list)

    return cp_mean


def consistency(forest_id, item, scale_factor=1000):
    crown_path = "./10-forest/forest%d/crown/%d.npy" % (forest_id, item)
    graph_path = "./Ours_Result/forest%d/graph/%d.npy" % (forest_id, item)
    crown_xyz, max_z = load_crown(crown_path)
    data = load_graph(graph_path, max_z)
    xyz = data[:, :3]

    consistency_ = square_distance(torch.Tensor(crown_xyz).unsqueeze(0), torch.Tensor(xyz).unsqueeze(0))[0].min(dim=1)[0].numpy()
    return np.mean(consistency_) * scale_factor


if __name__ == '__main__':
    clip_model = get_clip_model("RN50x16")
    clip_model.eval()
    cp = CLIP_Precision(forest_id=6, item=76, clip_model=clip_model)
    print("clip precision: %.5f" % cp)

    consist = consistency(forest_id=6, item=76)
    print("consistency: %.5f" % consist)