
from typing import Callable, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import to_2tuple

from SaliencyI2PLoc.modules.sphereModel import SphereConv2d as Sphere_Conv2d
from utils.logger import print_log

class PatchEmbed(nn.Module):
    """ 
        2D Image to Patch Embedding, original from TIMM, remove output_fmt related functions here
    """
    def __init__(
            self,
            img_size: Union[int, Tuple[int, int]] = 224,
            patch_size: Union[int, Tuple[int, int]] = 16,
            in_chans: int = 3,
            embed_dim: int = 768,
            norm_layer: Optional[Callable] = None,
            flatten: bool = True,
            output_fmt: Optional[str] = None,
            bias: bool = True,
            strict_img_size: bool = True, # support arbitrary input resolution
    ):
        super().__init__()
        self.patch_size = to_2tuple(patch_size) # to_2tuple handles of the input is already tuple
        if img_size is not None:
            self.img_size = to_2tuple(img_size)
            self.grid_size = tuple([s // p for s, p in zip(self.img_size, self.patch_size)])
            self.num_patches = self.grid_size[0] * self.grid_size[1]
        else:
            self.img_size = None
            self.grid_size = None
            self.num_patches = None

        if output_fmt is not None:
            self.flatten = False
        else:
            # flatten spatial dim and transpose to channels last, kept for bwd compat
            self.flatten = flatten
        self.strict_img_size = strict_img_size

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size, bias=bias)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        if self.img_size is not None:
            if self.strict_img_size:
                if H != self.img_size[0]:
                    print_log(f"Input height ({H}) doesn't match model ({self.img_size[0]}).")
                if W != self.img_size[1]:
                    print_log(f"Input width ({W}) doesn't match model ({self.img_size[1]}).")
            else:
                if H % self.patch_size[0] != 0:
                    print_log(f"Input height ({H}) should be divisible by patch size ({self.patch_size[0]}).")
                if W % self.patch_size[1] != 0:
                    print_log(f"Input width ({W}) should be divisible by patch size ({self.patch_size[1]}).")

        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # NCHW -> NLC
        x = self.norm(x)
        return x


class ConvEmbed(nn.Module):
    """ 
        Image to Patch Embedding with convolution
        Input: C,H,W
        Output: N,C
    """
    def __init__(self, img_size: Union[int, Tuple[int, int]] = 224,
                 patch_size: Union[int, Tuple[int, int]] = 16,
                 in_chans=3, embed_dim=768, norm_layer=None, flatten=True, bias: bool = True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        # self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.num_patches = 196
        self.flatten = flatten
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=3, stride=(2,2), padding=1),# [512,1024,3] -> [256,512,64]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=(2,2), padding=1),# [256,512,64] -> [128,256,128]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=(2,2), padding=1),# [128,256,128] -> [64,128,128]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=(2,2), padding=1),# [64,128,128] -> [32,64,256]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=(2,2), padding=1),# [32,64,256] -> [16,32,256]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=(1,2), padding=1),# [16,32,256] -> [16,16,512]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, embed_dim, kernel_size=1, stride=(1,1), padding=0)# [16,16,512] -> [16,16,768]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((14, 14))
        
    def forward(self, x):      
        B, C, H, W = x.shape
        x = self.conv_stem(x)
        x = self.avgpool(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        x = F.normalize(x, dim=-1)
        return x


class SphereConvEmbed_384(nn.Module):
    """ 
        Image to Patch Embedding with convolution
        Input: C,H,W
        Output: N,C
    """
    def __init__(self, img_size: Union[int, Tuple[int, int]] = 224,
                 patch_size: Union[int, Tuple[int, int]] = 16,
                 in_chans=3, embed_dim=384, norm_layer=None, flatten=True, bias: bool = True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        self.flatten = flatten
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.conv_stem = nn.Sequential(
            Sphere_Conv2d(in_chans, 64, kernel_size=3, stride=(2,2), padding=1, bias=False), # [512,1024,3] -> [256,512,64]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Sphere_Conv2d(64, 128, kernel_size=3, stride=(2,2), padding=1, bias=False), # [256,512,64] -> [128,256,128]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Sphere_Conv2d(128, 128, kernel_size=3, stride=(2,2), padding=1, bias=False), # [128,256,128] -> [64,128,128]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Sphere_Conv2d(128, 256, kernel_size=3, stride=(2,2), padding=1, bias=False), # [64,128,128] -> [32,64,256]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            Sphere_Conv2d(256, 256, kernel_size=3, stride=(2,2), padding=1, bias=False), # [32,64,256] -> [16,32,256]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            Sphere_Conv2d(256, embed_dim, kernel_size=3, stride=(1,2), padding=0, bias=False), # [16,16,256] -> [16,16,384]
        )
        self.num_patches = 196 # changed with the cnn feature map dismension
        self.avgpool = nn.AdaptiveAvgPool2d((14, 14)) # 196 patches
        # self.avgpool = SphereMaxPool2d((32, 32)) # 1024 patches

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv_stem(x)
        x = self.avgpool(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        x = F.normalize(x, dim=-1)
        return x


class SphereConvEmbed_768(nn.Module):
    """ 
        Image to Patch Embedding with convolution
        Input: C,H,W
        Output: N,C
    """
    def __init__(self, img_size: Union[int, Tuple[int, int]] = 224,
                 patch_size: Union[int, Tuple[int, int]] = 16,
                 in_chans=3, embed_dim=768, norm_layer=None, flatten=True, bias: bool = True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])

        self.flatten = flatten
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.conv_stem = nn.Sequential(
            Sphere_Conv2d(in_chans, 64, kernel_size=3, stride=(2,2), padding=1, bias=False), # [512,1024,3] -> [256,512,64]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            Sphere_Conv2d(64, 128, kernel_size=3, stride=(2,2), padding=1, bias=False), # [256,512,64] -> [128,256,128]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Sphere_Conv2d(128, 128, kernel_size=3, stride=(2,2), padding=1, bias=False), # [128,256,128] -> [64,128,128]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            Sphere_Conv2d(128, 256, kernel_size=3, stride=(2,2), padding=1, bias=False), # [64,128,128] -> [32,64,256]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            Sphere_Conv2d(256, 256, kernel_size=3, stride=(2,2), padding=1, bias=False), # [32,64,256] -> [16,32,256]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            Sphere_Conv2d(256, embed_dim, kernel_size=1, stride=(2,2), padding=0, bias=False), # [16,16,512] -> [16,16,768]
        )
        self.num_patches = 196 # changed with the cnn feature map dismension
        self.avgpool = nn.AdaptiveAvgPool2d((14, 14)) # 196 patches
        # self.avgpool = SphereMaxPool2d((32, 32)) # 1024 patches

    def forward(self, x):
        B, C, H, W = x.shape
        x = self.conv_stem(x)
        x = self.avgpool(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        x = F.normalize(x, dim=-1)
        return x


class ConvEmbed_backup(nn.Module):
    """ 
        Image to Patch Embedding with more convolution, inspired from SAIG, the convolution result is same with the PatchEmbed
    """
    def __init__(self, img_size: Union[int, Tuple[int, int]] = 224,
                 patch_size: Union[int, Tuple[int, int]] = 16,
                 in_chans=3, embed_dim=768, norm_layer=None, flatten=True, bias: bool = True,):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = (img_size[0] // patch_size[0], img_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=3, stride=2, padding=1),# [224,224,3] -> [112,112,64]
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),# [112,112,64] -> [56,56,128]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),# [56,56,128] -> [56,56,128]
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),# [56,56,128] -> [28,28,256]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),# [28,28,256] -> [28,28,256]
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),# [28,28,256] -> [14,14,512]
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, embed_dim, kernel_size=1, stride=1, padding=0)# [14,14,512] -> [14,14,768]
        )
        
    def forward(self, x):      
        B, C, H, W = x.shape
        x = self.conv_stem(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class MiniEncoder(nn.Module):
    '''
        MiniEncoder for mapping the input image to a fixed feature map, e.g. 224x224
        Worth to note that: the feature dimension is too large given the limited memory
    '''
    def __init__(self, in_chans:int=3, out_feat_size:int=224, embed_dim:int=768, norm_layer=nn.BatchNorm2d, bias:bool=True):
        super(MiniEncoder, self).__init__()
        out_feat_size = to_2tuple(out_feat_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        self.conv_stem = nn.Sequential(
            nn.Conv2d(in_chans, 64, kernel_size=3, stride=(1,1), padding=1, bias=bias),# [512,1024,3] -> [512,1024,64]
            norm_layer(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=(2,2), padding=1, bias=bias),# [512,1024,64] -> [256,512,128]
            norm_layer(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=(1,2), padding=1, bias=bias),# [256,512,128] -> [256,256,256]
            norm_layer(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 512, kernel_size=3, stride=(1,1), padding=1, bias=bias),# [256,256,256] -> [256,256,512]
            norm_layer(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, embed_dim, kernel_size=1, stride=(1,1), padding=0, bias=bias)# [256,256,512] -> [256,256,768]
        )
        self.avgpool = nn.AdaptiveAvgPool2d(out_feat_size)
        self._init_parameters()

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.avgpool(x)
        x = self.norm(x)
        return x            
                    
    def _init_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) \
                    or isinstance(m, nn.BatchNorm1d) \
                    or isinstance(m, nn.BatchNorm3d) \
                    or isinstance(m, nn.InstanceNorm2d) \
                    or isinstance(m, nn.InstanceNorm1d) \
                    or isinstance(m, nn.InstanceNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class HybridEmbed(nn.Module):
    """ 
        CNN Feature Map Embedding
        Extract feature map from CNN, flatten, project to embedding dim.
    """
    def __init__(self, backbone, img_size=224, patch_size=1, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # NOTE Most reliable way of determining output dims is to run forward pass
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    o = o[-1]  # last feature if backbone outputs list/tuple of features
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        assert feature_size[0] % patch_size[0] == 0 and feature_size[1] % patch_size[1] == 0
        self.grid_size = (feature_size[0] // patch_size[0], feature_size[1] // patch_size[1])
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[-1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2) # NCHW -> NLC
        return x


if __name__ == '__main__':
    '''
        Unit test of patch embedding
    '''
    input_tensor = torch.randn(2, 3, 512, 1024)
    # emb_layer1 = PatchEmbed(img_size=[512, 1024])
    # output1 = emb_layer1(input_tensor)
    # print(output1.size()) # torch.Size([2, 2048, 768])
    # emb_layer2 = ConvEmbed(img_size=[512, 1024])
    # output2 = emb_layer2(input_tensor)
    # print(output2.size()) # torch.Size([2, 196, 768])
    # emb_layer3 = HybridEmbed(backbone=MiniEncoder())
    # output3 = emb_layer3(input_tensor)
    # print(output3.size()) # torch.Size([2, 50176, 768])
    emb_layer4 = SphereConvEmbed_384(img_size=[512, 1024])
    output4 = emb_layer4(input_tensor)
    print(output4.size())
    emb_layer5 = SphereConvEmbed_768(img_size=[512, 1024])
    output5 = emb_layer5(input_tensor)
    print(output5.size())