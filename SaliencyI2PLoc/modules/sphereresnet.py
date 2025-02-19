import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from SaliencyI2PLoc.build import MODELS

from .sphereModel import SphereConv2d as Sphere_Conv2d
from .sphereModel import SphereMaxPool2d

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152']

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}


def sphereconv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return Sphere_Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class SphereBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SphereBasicBlock, self).__init__()
        self.conv1 = sphereconv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = sphereconv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SphereBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(SphereBottleneck, self).__init__()
        self.conv1 = Sphere_Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = Sphere_Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = Sphere_Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class SphereResNet(nn.Module):
    """
        Sphere-cnn based resnet architecture, without classification head
    """
    def __init__(self, in_channels, block, layers, num_classes=1000):
        self.inplanes = 64
        super(SphereResNet, self).__init__()
        self.conv1 = Sphere_Conv2d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = SphereMaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                Sphere_Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x
    
    def prune(self, sparsity, create_mask = True, prune_bias = False):
        first_layer = True
        if create_mask:
            self.masks = []
        masksid = -1
        for m in self.modules():
            if isinstance(m, Sphere_Conv2d) or isinstance(m, nn.Linear):
                if first_layer:
                    first_layer = False
                else:
                    if create_mask:
                        prune_k = int(m.weight.numel()*(sparsity))
                        val, _ = m.weight.view(-1).abs().topk(prune_k, largest = False)
                        prune_threshold = val[-1]
                        self.masks.append((m.weight.abs()<=prune_threshold).data)
                    masksid += 1
                    m.weight.data.masked_fill_(self.masks[masksid], 0) 
                    if prune_bias:
                        if create_mask:
                            prune_k = int(m.bias.numel()*(sparsity))
                            val, _ = m.bias.view(-1).abs().topk(prune_k, largest = False)
                            prune_threshold = val[-1]
                            self.masks.append((m.bias.abs()<=prune_threshold).data)
                        masksid += 1
                        m.bias.data.masked_fill_(self.masks[masksid], 0) 


def sphere_resnet18(in_channels=3, pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SphereResNet(in_channels, SphereBasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        pth = model_zoo.load_url(model_urls['resnet18'], progress=progress)
        model.load_state_dict(pth, strict=False)
    return model


def sphere_resnet34(in_channels=3, pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-34 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SphereResNet(in_channels, SphereBasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pth = model_zoo.load_url(model_urls['resnet34'], progress=progress)
        model.load_state_dict(pth, strict=False)
    return model


def sphere_resnet50(in_channels=3, pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-50 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SphereResNet(in_channels, SphereBottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pth = model_zoo.load_url(model_urls['resnet50'], progress=progress)
        model.load_state_dict(pth, strict=False)
    return model


def sphere_resnet101(in_channels=3, pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-101 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SphereResNet(in_channels, SphereBottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        pth = model_zoo.load_url(model_urls['resnet101'], progress=progress)
        model.load_state_dict(pth, strict=False)
    return model


def sphere_resnet152(in_channels=3, pretrained=False, progress=True, **kwargs):
    """Constructs a ResNet-152 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = SphereResNet(in_channels, SphereBottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        pth = model_zoo.load_url(model_urls['resnet152'], progress=progress)
        model.load_state_dict(pth, strict=False)
    return model


@MODELS.register_module()
class SphereResNetEncoder(SphereResNet):
    @classmethod
    def from_config(cls, config, from_pretrained=False):
        arch_type = config.arch_type
        if arch_type == "sphere_resnet18":
            encoder = sphere_resnet18(in_channels=3, pretrained=from_pretrained, progress=True)
        elif arch_type == "sphere_resnet34":
            encoder = sphere_resnet34(in_channels=3, pretrained=from_pretrained, progress=True)
        elif arch_type == "sphere_resnet50":
            encoder = sphere_resnet50(in_channels=3, pretrained=from_pretrained, progress=True)
        elif arch_type == "sphere_resnet101":
            encoder = sphere_resnet101(in_channels=3, pretrained=from_pretrained, progress=True)
        elif arch_type == "sphere_resnet152":
            encoder = sphere_resnet152(in_channels=3, pretrained=from_pretrained, progress=True)
        else:
            encoder = sphere_resnet34(in_channels=3, pretrained=from_pretrained, progress=True)      
        return encoder
            
    def forward_features(self, x, register_blk=-1):
        return super().forward(x, register_blk)