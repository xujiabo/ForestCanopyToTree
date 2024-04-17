import torch
from torch import nn
from utils import square_distance
from models.transformer import MHAttention, PositionEmbeddingCoordsSine


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda:0')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature  # (batch_size, 2*num_dims, num_points, k)


class DGCNN(nn.Module):
    def __init__(self):
        super(DGCNN, self).__init__()
        self.k = 30

        self.bn1 = nn.GroupNorm(32, 64)
        self.bn2 = nn.GroupNorm(32, 64)
        self.bn3 = nn.GroupNorm(32, 64)
        self.bn4 = nn.GroupNorm(32, 64)
        self.bn5 = nn.GroupNorm(32, 64)
        self.bn6 = nn.GroupNorm(32, 1024)
        self.bn7 = nn.GroupNorm(32, 64)
        self.bn8 = nn.GroupNorm(32, 256)
        self.bn9 = nn.GroupNorm(32, 256)
        self.bn10 = nn.GroupNorm(32, 128)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(192, 1024, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv7 = nn.Sequential(nn.Conv1d(16, 64, kernel_size=1, bias=False),
                                   self.bn7,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv8 = nn.Sequential(nn.Conv1d(1280, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(0.)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(0.)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                    self.bn10,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Conv1d(128, 256, kernel_size=1, bias=False)

    def forward(self, x):
        x = x.permute([0, 2, 1])
        batch_size = x.size(0)
        num_points = x.size(2)

        x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        # t = self.transform_net(x0)  # (batch_size, 3, 3)
        # x = x.transpose(2, 1)  # (batch_size, 3, num_points) -> (batch_size, num_points, 3)
        # x = torch.bmm(x, t)  # (batch_size, num_points, 3) * (batch_size, 3, 3) -> (batch_size, num_points, 3)
        # x = x.transpose(2, 1)  # (batch_size, num_points, 3) -> (batch_size, 3, num_points)

        # x = get_graph_feature(x, k=self.k)  # (batch_size, 3, num_points) -> (batch_size, 3*2, num_points, k)
        x = self.conv1(x)  # (batch_size, 3*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv2(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x1 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x1, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv3(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x = self.conv4(x)  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points, k)
        x2 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = get_graph_feature(x2, k=self.k)  # (batch_size, 64, num_points) -> (batch_size, 64*2, num_points, k)
        x = self.conv5(x)  # (batch_size, 64*2, num_points, k) -> (batch_size, 64, num_points, k)
        x3 = x.max(dim=-1, keepdim=False)[0]  # (batch_size, 64, num_points, k) -> (batch_size, 64, num_points)

        x = torch.cat((x1, x2, x3), dim=1)  # (batch_size, 64*3, num_points)

        x = self.conv6(x)  # (batch_size, 64*3, num_points) -> (batch_size, emb_dims, num_points)
        x = x.max(dim=-1, keepdim=True)[0]  # (batch_size, emb_dims, num_points) -> (batch_size, emb_dims, 1)

        # l = l.view(batch_size, -1, 1)  # (batch_size, num_categoties, 1)
        # l = self.conv7(l)  # (batch_size, num_categoties, 1) -> (batch_size, 64, 1)
        l = torch.zeros(x.shape[0], 64, 1).to(x.device)

        x = torch.cat((x, l), dim=1)  # (batch_size, 1088, 1)
        x = x.repeat(1, 1, num_points)  # (batch_size, 1088, num_points)

        x = torch.cat((x, x1, x2, x3), dim=1)  # (batch_size, 1088+64*3, num_points)

        x = self.conv8(x)  # (batch_size, 1088+64*3, num_points) -> (batch_size, 256, num_points)
        x = self.dp1(x)
        x = self.conv9(x)  # (batch_size, 256, num_points) -> (batch_size, 256, num_points)
        x = self.dp2(x)
        x = self.conv10(x)  # (batch_size, 256, num_points) -> (batch_size, 128, num_points)
        x = self.conv11(x)  # (batch_size, 256, num_points) -> (batch_size, seg_num_all, num_points)

        x = x.permute([0, 2, 1])
        return x


class PointTransformer(nn.Module):
    def __init__(self, token_dim=256, self_layers=8):
        super(PointTransformer, self).__init__()
        self.embedding = nn.Conv1d(3, token_dim, kernel_size=1, stride=1)
        self.pre_norm = nn.LayerNorm(token_dim)

        self.pe_func = PositionEmbeddingCoordsSine(n_dim=3, d_model=token_dim)
        self.self_layer_num = self_layers

        self.stem = nn.ModuleList()
        for i in range(self_layers):
            self.stem.append(MHAttention(token_dim, nhead=8))

    def forward(self, x):
        feats = self.pe_func(x) + self.embedding(x.permute([0, 2, 1])).permute([0, 2, 1])
        feats = self.pre_norm(feats)
        for i in range(self.self_layer_num):
            feats = self.stem[i](feats, feats, feats)
        return feats


class GNN(nn.Module):
    def __init__(self):
        super(GNN, self).__init__()
        self.gcn = DGCNN()
        # self.gcn = PointTransformer()

        self.cos_fc = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, kernel_size=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=1),
        )

        self.Q = nn.Conv1d(128, 128, kernel_size=1)
        self.K = nn.Conv1d(128, 128, kernel_size=1)
        self.scale = 128 ** -0.5

        self.fc = nn.Sequential(
            nn.Conv1d(256, 256, kernel_size=1),
            nn.GroupNorm(32, 256),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 128, kernel_size=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 128, kernel_size=1),
            nn.GroupNorm(32, 128),
            nn.ReLU(inplace=True),
            # w + isl + ls
            nn.Conv1d(128, 3, kernel_size=1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # batch x n x 3
        # 加一个[0, 0, 0]
        batch = x.shape[0]
        # x = torch.cat([torch.zeros(batch, 1, 3), x], dim=1)
        feats = self.gcn(x)
        # btch x n x 128
        cos_feats = self.cos_fc(feats.permute([0, 2, 1]))
        q = self.Q(cos_feats).permute([0, 2, 1])
        k = self.K(cos_feats).permute([0, 2, 1])
        fa = torch.matmul(q, k.permute([0, 2, 1]))  # * self.scale

        # batch x n x n
        # print(torch.sqrt(square_distance(x, x))[0])
        # mask = (torch.sqrt(square_distance(x, x)+1e-8) < 0.5).int()
        # print(mask[0, 1].sum(0).item())
        mask = 1 - torch.eye(x.shape[1]).to(x.device).unsqueeze(0).repeat(x.shape[0], 1, 1)
        # print(mask)
        mask = (1 - mask) * (-1e8)
        fa = fa + mask
        # fa = fa.softmax(dim=-1)
        # fa = fa[:, 1:, :]

        y = self.fc(feats.permute([0, 2, 1])).permute([0, 2, 1])
        # y = self.fc(feats[:, 1:, :])
        width, is_leaf, leaf_size = y[:, :, 0], y[:, :, 1], y[:, :, 2]
        is_leaf = self.sigmoid(is_leaf)

        return fa, width, is_leaf, leaf_size