
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data

from SaliencyI2PLoc.aggregate_layers import NetVLADLoupe
from SaliencyI2PLoc.attention_utils import SEAttention
from SaliencyI2PLoc.build import MODELS


class Flatten(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

    def forward(self, input):
        return input.view(input.size(0), -1)


class STN3d(nn.Module):
    def __init__(self, num_points=2500, k=3, use_bn=True):
        super(STN3d, self).__init__()
        self.k = k
        self.kernel_size = 3 if k == 3 else 1
        self.channels = 1 if k == 3 else k
        self.num_points = num_points
        self.use_bn = use_bn
        self.conv1 = torch.nn.Conv2d(self.channels, 64, (1, self.kernel_size))
        self.conv2 = torch.nn.Conv2d(64, 128, (1,1))
        self.conv3 = torch.nn.Conv2d(128, 1024, (1,1))
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k*k)
        self.fc3.weight.data.zero_()
        self.fc3.bias.data.zero_()
        self.relu = nn.ReLU()

        if use_bn:
            self.bn1 = nn.BatchNorm2d(64)
            self.bn2 = nn.BatchNorm2d(128)
            self.bn3 = nn.BatchNorm2d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        batchsize = x.size()[0]
        if self.use_bn:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
        x = self.mp1(x)
        x = x.view(-1, 1024)

        if self.use_bn:
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        x = self.fc3(x)

        iden = torch.autograd.Variable(torch.from_numpy(np.eye(self.k).astype(np.float32))).view(
            1, self.k*self.k).repeat(batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class PointNet(nn.Module):
    """
        Standard PointNet encoder, project xyz into D high dimension feature space
    """
    def __init__(self, num_points=2500, global_feat=True, feature_transform=False, max_pool=True):
        super(PointNet, self).__init__()
        self.stn = STN3d(num_points=num_points, k=3, use_bn=False)
        self.feature_trans = STN3d(num_points=num_points, k=64, use_bn=False)
        self.apply_feature_trans = feature_transform
        self.conv1 = torch.nn.Conv2d(1, 64, (1, 3))
        self.conv2 = torch.nn.Conv2d(64, 64, (1, 1))
        self.conv3 = torch.nn.Conv2d(64, 64, (1, 1))
        self.conv4 = torch.nn.Conv2d(64, 128, (1, 1))
        self.conv5 = torch.nn.Conv2d(128, 1024, (1, 1))
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)
        self.bn5 = nn.BatchNorm2d(1024)
        self.mp1 = torch.nn.MaxPool2d((num_points, 1), 1)
        self.num_points = num_points
        self.global_feat = global_feat
        self.max_pool = max_pool

    def forward(self, x):
        batchsize = x.size()[0]
        trans = self.stn(x)
        x = torch.matmul(torch.squeeze(x), trans) # BNC \times 3x3
        x = x.view(batchsize, 1, -1, 3) # BNC -> B1NC
        x = F.relu(self.bn1(self.conv1(x))) # B1NC -> B64N1
        x = F.relu(self.bn2(self.conv2(x))) # B64N1 -> B64N1
        pointfeat = x
        if self.apply_feature_trans:
            f_trans = self.feature_trans(x) # B64x64
            x = torch.squeeze(x) # B64N1 -> B64N
            if batchsize == 1:
                x = torch.unsqueeze(x, 0)
            x = torch.matmul(x.transpose(1, 2), f_trans) #B64N -> BN64 \times 64x64 -> BN64
            x = x.transpose(1, 2).contiguous() # B64N
            x = x.view(batchsize, 64, -1, 1) # B64N -> B64N1
        x = F.relu(self.bn3(self.conv3(x))) # B64N1 -> B64N1
        x = F.relu(self.bn4(self.conv4(x))) # B64N1 -> B128N1
        x = self.bn5(self.conv5(x)) # B128N1 -> B1024N1
        if not self.max_pool: # global feature
            return x
        else:
            x = self.mp1(x)
            x = x.view(-1, 1024)
            if self.global_feat:
                return x, trans
            else:
                x = x.view(-1, 1024, 1).repeat(1, 1, self.num_points)
                return torch.cat([x, pointfeat], 1), trans


class PointNetVlad_attention(nn.Module):
    def __init__(self, num_points=4096, global_feat=True, feature_transform=True, max_pool=True, output_dim=256, num_clusters=64):
        super(PointNetVlad_attention, self).__init__()
        self.point_net = PointNet(num_points=num_points, global_feat=global_feat,
                                      feature_transform=feature_transform, max_pool=max_pool)
        self.attention = SEAttention(channel=1024, reduction=8) # 1024 is feature dimension after PointNet for each point
        config = dict(feature_size=1024,
                      num_clusters=num_clusters, output_dim=output_dim,
                      gating=True, add_batch_norm=True, work_with_tokens=False)
        self.net_vlad = NetVLADLoupe(config)
        self._init_weight_()

    def _init_weight_(self):
        self.attention.init_weights()

    def forward(self, x):
        x = self.point_net(x) # B1N3 -> B1024N1
        x = self.attention(x) # B1024N1 -> B1024N1
        x = self.net_vlad(x) # B1024N -> B256
        return x
    

@MODELS.register_module()
class PointNetVLADEncoder(nn.Module):
    """
        The pointnet based point cloud encoder combined with a se-attention and vlad layer 
    """
    @classmethod
    def from_config(cls, config):
        pc_encoder = PointNetVlad_attention(num_points=config.num_points, global_feat=True, feature_transform=True, 
                                            max_pool=False, output_dim=config.global_feat_dim, 
                                            num_clusters=config.num_clusters)
        pc_encoder.embed_dim = config.global_feat_dim
        return pc_encoder

    def forward(self, x):
        """foward

        Args:
            x (torch:tensor): the input tensor

        Returns:
            embedding: the output features after encoding 
        """
        return self.pc_encoder(x)


if __name__ == '__main__':
    num_points = 4096
    sim_data = torch.autograd.Variable(torch.rand(4, 1, num_points, 3))
    sim_data = sim_data.cuda()

    pnv = PointNetVlad_attention(global_feat=True, feature_transform=True, max_pool=False,
                                    output_dim=256, num_points=num_points).cuda()
    pnv.train()
    out3 = pnv(sim_data)
    print('pnv', out3.size())