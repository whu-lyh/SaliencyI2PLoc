
import torch.nn as nn
import torchvision.models as models

from SaliencyI2PLoc.aggregate_layers import NetVLAD
from SaliencyI2PLoc.attention_utils import SEAttention
from SaliencyI2PLoc.build import MODELS
from SaliencyI2PLoc.modules.sphereresnet import sphere_resnet18, sphere_resnet34


def resnet18(pretrained=True):
    enc = models.resnet18(pretrained=pretrained)
    layers = list(enc.children())[:-2]
    index = -1
    # resnet: only train the last module(layer[7])
    for layer in layers[:index]:
        for p in layer.parameters():
            p.requires_grad = False
    enc = nn.Sequential(*layers)
    return enc


def resnet34(pretrained=True):
    enc = models.resnet34(pretrained=pretrained)
    layers = list(enc.children())[:-2]
    index = -1
    # resnet: only train the last module(layer[7])
    for layer in layers[:index]:
        for p in layer.parameters():
            p.requires_grad = False
    enc = nn.Sequential(*layers)
    return enc


class SphereNetVlad_attention(nn.Module):
    def __init__(self, backbone, encoder_dim:int=512, global_feat_dim:int=256, num_clusters:int=64):
        """
            here 512(=16x32) is set by default to cover resnet18 and resnet34 simultaneously
            channels = 512 if name in ('resnet18', 'resnet34') else 2048
        """
        super(SphereNetVlad_attention, self).__init__()
        self.sphere_net = backbone
        self.attention = SEAttention(channel=encoder_dim, reduction=8)
        config = dict(num_clusters=num_clusters, 
                      dim=encoder_dim, 
                      out_dim=global_feat_dim,
                      alpha=1.0,
                      work_with_tokens=False)
        self.net_vlad = NetVLAD(config)

        self._init_weight_()

    def _init_weight_(self):
        self.attention.init_weights()

    def forward(self, x):
        x = self.sphere_net(x) # B C H W -> B C2 H2 W2
        x = self.attention(x) # B C2 H2 W2 -> B C2 H2 W2
        x = self.net_vlad(x) # B C2 H2 W2 -> B 256
        return x


@MODELS.register_module()
class SphereNetVLADEncoder(nn.Module):
    """
        The sphere cnn based image encoder combined with a vlad layer 
    """
    @classmethod
    def from_config(cls, config):
        # get cnn stem
        encoder_type = config.encoder_type
        if encoder_type == "sphere_resnet18":
            backbone = sphere_resnet18(pretrained=config.from_pretrained)
        elif encoder_type == "sphere_resnet34":
            backbone = sphere_resnet34(pretrained=config.from_pretrained)
        elif encoder_type == "resnet18":
            backbone = resnet18(pretrained=config.from_pretrained)
        elif encoder_type == "resnet34":
            backbone = resnet34(pretrained=config.from_pretrained)
        else:
            raise NotImplementedError(f'Sorry, <{encoder_type}> encoder is not implemented!')
        # get aggregate stem
        visual_encoder = SphereNetVlad_attention(backbone, config.embed_dim, config.global_feat_dim, config.num_clusters)
        visual_encoder.embed_dim = config.global_feat_dim
        return visual_encoder

    def _freeze_stages(self):
        self.visual_encoder.eval()
        for param in self.visual_encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        """foward

        Args:
            x (torch:tensor): the input tensor

        Returns:
            embedding: the output features after encoding 
        """
        return self.visual_encoder(x)
