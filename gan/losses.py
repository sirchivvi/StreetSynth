import torch
import torch.nn as nn
import torchvision.models as models

class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).features
        self.blocks = nn.ModuleList([
            vgg[:4].eval(), vgg[4:9].eval(), vgg[9:16].eval()
        ])
        for p in self.parameters(): p.requires_grad = False
    def forward(self, pred, target):
        pred   = (pred   + 1) / 2
        target = (target + 1) / 2
        loss = 0.0
        x, y = pred, target
        for block in self.blocks:
            x = block(x); y = block(y)
            loss += nn.functional.l1_loss(x, y)
        return loss

def depth_consistency_loss(pred, depth_map):
    pred_gray = pred.mean(dim=1, keepdim=True)
    dy_pred   = torch.abs(pred_gray[:,:,1:,:] - pred_gray[:,:,:-1,:])
    dx_pred   = torch.abs(pred_gray[:,:,:,1:] - pred_gray[:,:,:,:-1])
    dy_depth  = torch.abs(depth_map[:,:,1:,:] - depth_map[:,:,:-1,:])
    dx_depth  = torch.abs(depth_map[:,:,:,1:] - depth_map[:,:,:,:-1])
    return (dy_pred * torch.exp(-dy_depth)).mean() + \
           (dx_pred * torch.exp(-dx_depth)).mean()