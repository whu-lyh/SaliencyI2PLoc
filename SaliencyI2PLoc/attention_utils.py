
from torch import nn
from torch.nn import init

class SEAttention(nn.Module):
    """
        Details in: "Squeeze-and-Excitation Networks"
        Other implementations could be found here: https://github.com/moskomule/senet.pytorch/blob/master/senet/se_module.py  
    """
    def __init__(self, channel=512, reduction=16):
        super().__init__()
        # pooling along the channel dimention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # FC layer with gate operation
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c) # BCN1 -> BC
        y = self.fc(y).view(b, c, 1, 1) # BC -> BC11
        return x * y.expand_as(x)
    
