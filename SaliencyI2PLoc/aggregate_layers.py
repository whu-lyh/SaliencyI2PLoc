
import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from SaliencyI2PLoc.build import MODELS


class GatingContext(nn.Module):
    """
        Dimension reduction? # TODO
    """
    def __init__(self, dim, add_batch_norm=True):
        super(GatingContext, self).__init__()
        self.dim = dim
        self.add_batch_norm = add_batch_norm
        self.gating_weights = nn.Parameter(torch.randn(dim, dim) * 1 / math.sqrt(dim))
        self.sigmoid = nn.Sigmoid()

        if add_batch_norm:
            self.gating_biases = None
            self.bn1 = nn.BatchNorm1d(dim)
        else:
            self.gating_biases = nn.Parameter(
                torch.randn(dim) * 1 / math.sqrt(dim))
            self.bn1 = None

    def forward(self, x):
        gates = torch.matmul(x, self.gating_weights)

        if self.add_batch_norm:
            gates = self.bn1(gates)
        else:
            gates = gates + self.gating_biases

        gates = self.sigmoid(gates)
        activation = x * gates
        return activation


@MODELS.register_module()
class NetVLAD(nn.Module):
    """
        NetVLAD layer
        From paper "NetVLAD: CNN Architecture for Weakly Supervised Place Recognition"
        Here, the NetVLAD layer requires call init_params() firstly
    """
    def __init__(self, config):
        """
            Args:
                num_clusters : int, by default 64
                    The number of clusters.
                dim : int, by default 256
                    Dimension of input local descriptors.
                out_dim : int, by default 256
                    Dimension of output descriptors.
                alpha : float, by default 1.0
                    Parameter of initialization. Larger value is harder assignment.
                work_with_tokens: bool, by default True
                    If true, the backbone is from vit model
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = config["num_clusters"]
        self.dim = config["dim"]
        self.out_dim = config["out_dim"]
        self.alpha = config["alpha"]
        self.normalize_input = True
        self.work_with_tokens = config["work_with_tokens"] # True if backbone is vit-based
        if self.work_with_tokens:
            self.conv = nn.Conv1d(self.dim, self.num_clusters, kernel_size=1, bias=False) # 1d conv will not change dimension
        else:
            self.conv = nn.Conv2d(self.dim, self.num_clusters, kernel_size=(1, 1), bias=False)
        # VLAD Core
        self.centroids = nn.Parameter(torch.rand(self.num_clusters, self.dim)) # K C(input feature dimension)
        # linear projection to final dim(D) dimension
        self.hidden1_weights = nn.Parameter(torch.randn(self.num_clusters * self.dim, self.out_dim) * 1 / math.sqrt(self.dim)) # K D(output feature dimension)
        # GatingContext
        self.bn2 = nn.BatchNorm1d(self.out_dim)
        self.context_gating = GatingContext(self.out_dim, add_batch_norm=True)

    def init_params(self, clsts, traindescs):
        clstsAssign = clsts / np.linalg.norm(clsts, axis=1, keepdims=True)
        dots = np.dot(clstsAssign, traindescs.T)
        dots.sort(0)
        dots = dots[::-1, :]  # sort, descending
        self.alpha = (-np.log(0.01) / np.mean(dots[0, :] - dots[1, :])).item()
        # noinspection PyArgumentList
        self.centroids = nn.Parameter(torch.from_numpy(clsts))
        # noinspection PyArgumentList
        if self.work_with_tokens:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * clstsAssign).unsqueeze(2))
        else:
            self.conv.weight = nn.Parameter(torch.from_numpy(self.alpha * clstsAssign).unsqueeze(2).unsqueeze(3))
        self.conv.bias = None

    def forward(self, x):
        """NetVLAD forward function
        here the C happen to be same as the NetVLAD feature dimension
        and the D is same as C which might be tricky for understanding

        Args:
            x (torch.Tensor): B C H W feature map or B N C tokens from transformer model

        Returns:
            torch.Tensor: B D, vlad global feature
        """
        if self.work_with_tokens:
            x = x.permute(0, 2, 1) # B N C -> B C N
            B, C, _ = x.shape[:]
        else:
            B, C = x.shape[:2] # B C H W
        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim # B C N
        # soft-assignment: assign weight to all centroids
        soft_assign = self.conv(x).view(B, self.num_clusters, -1) # B C N -> B K N -> B K N
        soft_assign = F.softmax(soft_assign, dim=1) # B K N, along the num_clusters dimension
        x_flatten = x.view(B, C, -1) # B C H W -> B C N or B C N -> B C N
        # calculate residuals to each clusters
        '''
        # B C N -> K B C N -> B K C N
        # K C -> N K C -> K C N -> 1 K C N
        # residual: B K C N - 1 K C N = B K C N
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
            self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2) # B K C N * B K 1 N = B K C N
        vlad = residual.sum(dim=-1) # B K C N -> B K C
        '''
        vlad = torch.zeros([B, self.num_clusters, C], dtype=x.dtype, layout=x.layout, device=x.device) # B K C
        for Ck in range(self.num_clusters):  # slower than non-looped, but lower memory usage
            # x_flatten.unsqueeze(0).permute(1, 0, 2, 3) # B C N -> 1 B C N -> B 1 C N
            # self.centroids[Ck:Ck + 1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            # 1 C -> N 1 C -> 1 C N -> 1 1 C N
            # B 1 C N - 1 1 C N = B 1 C N
            residual = x_flatten.unsqueeze(0).permute(1, 0, 2, 3) - \
                       self.centroids[Ck:Ck + 1, :].expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
            residual *= soft_assign[:, Ck:Ck + 1, :].unsqueeze(2) # B 1 C N * (B 1 N -> B 1 1 N) = B 1 C N
            # sum whole residuals from all local features
            vlad[:, Ck:Ck + 1, :] = residual.sum(dim=-1) # B 1 C = (B 1 C N -> B 1 C)

        vlad = F.normalize(vlad, p=2, dim=2) # intra-normalization, inside cluster # B K C
        vlad = vlad.view(B, -1) # flatten vlad # B K C -> B K*C
        vlad = F.normalize(vlad, p=2, dim=1) # L2-normalization # B K*C
        vlad = torch.matmul(vlad, self.hidden1_weights) # dimension reduction # B K*C -> B D
        vlad = self.bn2(vlad)
        vlad = self.context_gating(vlad) # TODO a kind of attention?
        return vlad


@MODELS.register_module()
class NetVLADLoupe(nn.Module):
    """
        NetVLAD layer
        From paper PointNetVLAD
    """
    def __init__(self, config):
        """
        Args:
            num_clusters : int, by default 64
                The number of clusters.
            feature_size : int, by default 256
                Dimension of input local descriptors.
            output_dim : int, by default 256
                Dimension of output descriptors.
            add_batch_norm : bool, by default False
                Do batch normalization
            gating : bool, by default True
                If true, descriptor-wise L2 normalization is applied to input.
            work_with_tokens: bool, by default True
                If true, the backbone is from vit model
        """
        super(NetVLADLoupe, self).__init__()
        self.feature_size = config["feature_size"]
        self.num_clusters = config["num_clusters"]
        self.output_dim = config["output_dim"]
        self.gating = config["gating"]
        self.add_batch_norm = config["add_batch_norm"]
        self.work_with_tokens = config["work_with_tokens"] # True if backbone is vit-based

        self.softmax = nn.Softmax(dim=-1)
        # make the vlad differentiable
        self.cluster_weights = nn.Parameter(torch.randn(self.feature_size, self.num_clusters) * 1 / math.sqrt(self.feature_size)) # C K
        # VLAD Core centroids that will be learnable parameters
        self.cluster_weights2 = nn.Parameter(torch.randn(1, self.feature_size, self.num_clusters) * 1 / math.sqrt(self.feature_size)) # 1 C K
        # GatingContext
        self.hidden1_weights = nn.Parameter(torch.randn(self.num_clusters * self.feature_size, self.output_dim) * 1 / math.sqrt(self.feature_size)) # K*C, output dimension

        if self.add_batch_norm:
            self.cluster_biases = None
            self.bn1 = nn.BatchNorm1d(self.num_clusters)
        else:
            self.cluster_biases = nn.Parameter(torch.randn(self.num_clusters) * 1 / math.sqrt(self.feature_size))
            self.bn1 = None

        self.bn2 = nn.BatchNorm1d(self.output_dim)
        if self.gating:
            self.context_gating = GatingContext(self.output_dim, add_batch_norm=self.add_batch_norm)

    def forward(self, x, token_weight=None):
        """NetVLAD forward function

        Args:
            x (torch.Tensor): input feature, size = BNC or BCN1
            token_weight (torch.Tensor, optional): token weights from attention. size = B, N, Defaults to None.

        Returns:
            vlad global feature (torch.Tensor): size = B D
        """
        if self.work_with_tokens: # from transformer token output
            x = x.permute(0, 2, 1).unsqueeze(3) # BNC -> BCN -> BCN1 # for be compatible with netvlad layer
        # make sure input x == B C N 1
        self.max_samples = x.shape[2]
        x = x.transpose(1, 3).contiguous() # B C N 1 -> B 1 N C
        x = x.view((-1, self.max_samples, self.feature_size)) # B 1 N C -> B*1 N C = B N C
        # a learnable parameter that indicate the confidience of each local belonging to each cluster
        activation = torch.matmul(x, self.cluster_weights) # B N C * C K = B N K

        if self.add_batch_norm:
            activation = activation.view(-1, self.num_clusters) # B N K -> B*N K
            activation = self.bn1(activation) # B*N K
            activation = activation.view(-1, self.max_samples, self.num_clusters) # B*N K -> B N K
        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation) # B N K
        if token_weight is not None:
            # poor performance
            # token_weight = F.normalize(token_weight, dim=-1)
            # activation = activation * token_weight.unsqueeze(-1) # B N K * B N = B N K
            # better performance
            token_weight = token_weight.reshape([-1, self.max_samples, 1])
            token_weight = torch.sigmoid(token_weight)
            token_weight = token_weight.repeat(1, 1, self.num_clusters)
            activation = torch.mul(activation, token_weight)

        activation = activation.view((-1, self.max_samples, self.num_clusters)) # B N K -> B N K
        # sum from whole N local features
        a_sum = activation.sum(-2, keepdim=True) # B 1 K
        a = a_sum * self.cluster_weights2 # B 1 K * 1 C K = B C K
        # same results between * and torch.mul # element-wise multiply, broadcast mechanism
        # a = torch.mul(a_sum, self.cluster_weights2)

        activation = torch.transpose(activation, 2, 1) # B N K -> B K N
        x = x.view((-1, self.max_samples, self.feature_size)) # B N C -> B N C
        vlad = torch.matmul(activation, x) # B K N * B N C = B K C
        vlad = torch.transpose(vlad, 2, 1) # B K C -> B C K
        vlad = vlad - a # B C K - B C K = B C K

        vlad = F.normalize(vlad, dim=1, p=2) # intra-normalization, inside cluster # B C K
        vlad = vlad.contiguous().view((-1, self.num_clusters * self.feature_size)) # flatten vlad # B C K -> B C*K
        vlad = F.normalize(vlad, dim=1, p=2) # L2-normalization # B C*K
        vlad = torch.matmul(vlad, self.hidden1_weights) # B C*K * C*K C_OUT = B C_OUT
        vlad = self.bn2(vlad)
        if self.gating:
            vlad = self.context_gating(vlad)
        return vlad

    def forward_vlad_residual(self, x, token_weight=None):
        """NetVLAD forward function that return the residual feature

        Args:
            x (torch.Tensor): input feature, size = BNC or BCN1
            token_weight (torch.Tensor, optional): token weights from attention. size = B, N, Defaults to None.

        Returns:
            vlad residual feature (torch.Tensor): size = B D K
        """
        if self.work_with_tokens: # from transformer token output
            x = x.permute(0, 2, 1).unsqueeze(3) # BNC -> BCN -> BCN1 # for be compatible with netvlad layer
        # make sure input x == B C N 1
        self.max_samples = x.shape[2]
        x = x.transpose(1, 3).contiguous() # B C N 1 -> B 1 N C
        x = x.view((-1, self.max_samples, self.feature_size)) # B 1 N C -> B*1 N C = B N C
        # a learnable parameter that indicate the confidience of each local belonging to each cluster
        activation = torch.matmul(x, self.cluster_weights) # B N C * C K = B N K

        if self.add_batch_norm:
            activation = activation.view(-1, self.num_clusters) # B N K -> B*N K
            activation = self.bn1(activation) # B*N K
            activation = activation.view(-1, self.max_samples, self.num_clusters) # B*N K -> B N K
        else:
            activation = activation + self.cluster_biases
        activation = self.softmax(activation) # B N K
        if token_weight is not None:
            # poor performance
            # token_weight = F.normalize(token_weight, dim=-1)
            # activation = activation * token_weight.unsqueeze(-1) # B N K * B N = B N K
            # better performance
            token_weight = token_weight.reshape([-1, self.max_samples, 1])
            token_weight = torch.sigmoid(token_weight)
            token_weight = token_weight.repeat(1, 1, self.num_clusters)
            activation = torch.mul(activation, token_weight)

        activation = activation.view((-1, self.max_samples, self.num_clusters)) # B N K -> B N K
        # sum from whole N local features
        a_sum = activation.sum(-2, keepdim=True) # B 1 K
        a = a_sum * self.cluster_weights2 # B 1 K * 1 C K = B C K
        # same results between * and torch.mul # element-wise multiply, broadcast mechanism
        # a = torch.mul(a_sum, self.cluster_weights2)

        activation = torch.transpose(activation, 2, 1) # B N K -> B K N
        x = x.view((-1, self.max_samples, self.feature_size)) # B N C -> B N C
        vlad = torch.matmul(activation, x) # B K N * B N C = B K C
        vlad = torch.transpose(vlad, 2, 1) # B K C -> B C K
        vlad = vlad - a # B C K - B C K = B C K
        vlad = F.normalize(vlad, dim=1, p=2) # intra-normalization, inside cluster # B C K
        return vlad

@MODELS.register_module()
class GeM(nn.Module):
    """
        Generalized-mean (GeM) pooling layer with one learnable parameter
        From "Fine-tuning CNN Image Retrieval with No Human Annotation"
        Return (B, C) tensor
    """
    def __init__(self, config):
        super(GeM,self).__init__()
        self.p = nn.Parameter(torch.ones(1)*config.p)
        self.eps = float(config.eps)

    def forward(self, x):
        return self.gem(x, p=self.p, eps=self.eps)

    def gem(self, x, p=3, eps=1e-6):
        return F.avg_pool2d(x.clamp(min=eps, max=1.0).pow(p), (x.size(-2), x.size(-1))).pow(1./p)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'


@MODELS.register_module()
class SMD(nn.Module):
    """
        Spatial-mixed feature aggregation module block used for feature pooling
        From "Simple, Effective and General: A New Backbone for Cross-view Image Geo-localization"
        Input: B, N, C
        Output: B, C
    """
    def __init__(self, config):
        super(SMD, self).__init__()
        in_dim = config.in_dim
        expand_dim = in_dim * config.expand_scale_factor
        squeeze_dim = in_dim
        k_scale_factor = config.k_scale_factor
        self.linear1 = nn.Linear(in_dim, expand_dim)
        self.act = nn.GELU()
        self.linear2 = nn.Linear(expand_dim, squeeze_dim)
        self.linear3 = nn.Linear(squeeze_dim, k_scale_factor)

    def forward(self, x):
        x = x.transpose(-1,-2) # [B N C] -> [B C N]
        x = self.linear1(x) # [B C N] -> [B C config.expand_dim]
        x = self.act(x)
        x = self.linear2(x) # [B C config.expand_dim] -> [B C config.squeeze_dim]
        x = self.linear3(x) # [B C config.squeeze_dim] -> [B C config.scale_dim]
        x = x.transpose(-1,-2).contiguous() # [B, config.scale_dim, C]
        x = x.flatten(-2,-1) # [B config.scale_dim x C]
        return x
