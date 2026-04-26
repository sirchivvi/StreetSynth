import torch
import torch.nn as nn

class PatchDiscriminator(nn.Module):
    def __init__(self, in_channels=8):
        super().__init__()
        def block(in_ch, out_ch, stride=2, norm=True):
            layers = [nn.Conv2d(in_ch, out_ch, 4, stride, 1, bias=False)]
            if norm: layers.append(nn.InstanceNorm2d(out_ch))
            layers.append(nn.LeakyReLU(0.2))
            return nn.Sequential(*layers)
        self.model = nn.Sequential(
            block(in_channels, 64,  norm=False),
            block(64,  128),
            block(128, 256),
            block(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, 1, 1)
        )
    def forward(self, inp, target):
        return self.model(torch.cat([inp, target], 1))

class MultiScaleDiscriminator(nn.Module):
    def __init__(self, in_channels=8):
        super().__init__()
        self.D1   = PatchDiscriminator(in_channels)
        self.D2   = PatchDiscriminator(in_channels)
        self.down = nn.AvgPool2d(2)
    def forward(self, inp, target):
        return self.D1(inp, target), self.D2(self.down(inp), self.down(target))