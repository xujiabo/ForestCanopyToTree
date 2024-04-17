import torch
from torch import nn
from torch.nn import functional as F
from models.transformer import PositionEmbeddingCoordsSine


class ResBlock(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(ResBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, out_dim)
        self.bn1 = nn.GroupNorm(32, out_dim)
        # self.bn = nn.BatchNorm1d(out_dim)
        self.relu = nn.GELU()
        self.fc2 = nn.Linear(out_dim, out_dim)
        self.bn2 = nn.GroupNorm(32, out_dim)
        self.ds = False
        if in_dim != out_dim:
            self.ds = True
            self.fc3 = nn.Linear(in_dim, out_dim)
            self.bn3 = nn.GroupNorm(32, out_dim)

    def forward(self, x):
        # x = self.relu(x + self.bn(self.fc(x)))
        sc = x if not self.ds else self.bn3(self.fc3(x))
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.bn2(self.fc2(x))
        x = self.relu(sc+x)
        return x


class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.conv1 = nn.Conv1d(3+64, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(128, 256, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(256, 512, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(32, 64)
        self.bn2 = nn.GroupNorm(32, 64)
        self.bn3 = nn.GroupNorm(32, 128)
        self.bn4 = nn.GroupNorm(32, 256)
        self.bn5 = nn.GroupNorm(32, 512)

        self.pe_func = PositionEmbeddingCoordsSine(3, 64)

    def forward(self, x):
        # B N 3
        batch = x.shape[0]
        # x = x * 100
        pe = self.pe_func(x)
        x = torch.cat([x, pe], dim=2)

        x = x.permute([0, 2, 1])
        x = F.gelu(self.bn1(self.conv1(x)))
        x = F.gelu(self.bn2(self.conv2(x)))
        x = F.gelu(self.bn3(self.conv3(x)))
        x = F.gelu(self.bn4(self.conv4(x)))
        x = F.gelu(self.bn5(self.conv5(x)))

        x = torch.cat([F.adaptive_max_pool1d(x, 1).view(batch, 512), F.adaptive_avg_pool1d(x, 1).view(batch, 512)], dim=1)

        return x

class PointFC(nn.Module):
    def __init__(self):
        super(PointFC, self).__init__()
        self.point_net = PointNet()
        self.fc = nn.Sequential(
            nn.Linear(1024, 4096),
            ResBlock(4096, 4096),
            nn.Linear(4096, 4096 * 3)
        )

    def forward(self, x):
        B = x.shape[0]
        f = self.point_net(x)
        generated_points = self.fc(f.view(B, 1024))
        return generated_points.view(B, 4096, 3)