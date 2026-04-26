import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetDown(nn.Module):
    def __init__(self, in_ch, out_ch, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False)]
        if normalize: layers.append(nn.InstanceNorm2d(out_ch))
        layers.append(nn.LeakyReLU(0.2))
        if dropout: layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)
    def forward(self, x): return self.block(x)

class UNetUp(nn.Module):
    def __init__(self, in_ch, out_ch, dropout=0.0):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_ch), nn.ReLU()
        ]
        if dropout: layers.append(nn.Dropout(dropout))
        self.block = nn.Sequential(*layers)
    def forward(self, x, skip):
        x = self.block(x)
        if x.shape != skip.shape:
            x = F.interpolate(x, size=skip.shape[2:],
                              mode="bilinear", align_corners=False)
        return torch.cat([x, skip], 1)

class AccessNetGenerator(nn.Module):
    def __init__(self, in_channels=5, out_channels=3, features=64):
        super().__init__()
        self.d1 = UNetDown(in_channels, features,   normalize=False)
        self.d2 = UNetDown(features,    features*2)
        self.d3 = UNetDown(features*2,  features*4)
        self.d4 = UNetDown(features*4,  features*8, dropout=0.5)
        self.d5 = UNetDown(features*8,  features*8, dropout=0.5)
        self.d6 = UNetDown(features*8,  features*8, dropout=0.5)
        self.d7 = UNetDown(features*8,  features*8, normalize=False)
        self.u1 = UNetUp(features*8,    features*8, dropout=0.5)
        self.u2 = UNetUp(features*16,   features*8, dropout=0.5)
        self.u3 = UNetUp(features*16,   features*8, dropout=0.5)
        self.u4 = UNetUp(features*16,   features*4)
        self.u5 = UNetUp(features*8,    features*2)
        self.u6 = UNetUp(features*4,    features)
        self.final = nn.Sequential(
            nn.ConvTranspose2d(features*2, out_channels, 4, 2, 1),
            nn.Tanh()
        )
    def forward(self, x):
        d1=self.d1(x); d2=self.d2(d1); d3=self.d3(d2); d4=self.d4(d3)
        d5=self.d5(d4); d6=self.d6(d5); d7=self.d7(d6)
        u1=self.u1(d7,d6); u2=self.u2(u1,d5); u3=self.u3(u2,d4)
        u4=self.u4(u3,d3); u5=self.u5(u4,d2); u6=self.u6(u5,d1)
        out = self.final(u6)
        if out.shape[2:] != x.shape[2:]:
            out = F.interpolate(out, size=x.shape[2:],
                                mode="bilinear", align_corners=False)
        return out