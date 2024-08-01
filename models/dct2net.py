import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def alpha(u):
    if u == 0:
        return 1 / np.sqrt(2)
    else:
        return 1


class DifferentiableThreshold(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):

        ctx.save_for_backward(x)
        m = 32

        # 近似する & nanを防ぐ
        th = 1.5
        mask = torch.abs(x) >= th

        y = torch.ones_like(x)
        z = torch.pow(x, 2*m)
        y[~mask] = z[~mask] / (z[~mask] + 1)

        return y
    
    @staticmethod
    def backward(ctx, dL_dy):

        x, = ctx.saved_tensors
        m = 32

        # 近似する & nanを防ぐ
        th = 1.5
        mask = torch.abs(x) >= th

        dy_dx = torch.zeros_like(x)
        z = torch.pow(x, 2*m-1)
        dy_dx[~mask] = 2 * m * z[~mask] / (x[~mask] * z[~mask] + 1) ** 2
        dL_dx = dL_dy * dy_dx

        return dL_dx


class DCT2Net(nn.Module):

    def __init__(self, patch_size, threshold):

        super(DCT2Net, self).__init__()

        self.patch_size = patch_size
        self.threshold = threshold

        P = np.zeros((patch_size**2, patch_size**2))
        for x in range(patch_size):
            for y in range(patch_size):
                for u in range(patch_size):
                    for v in range(patch_size):
                        P[x*patch_size+y, u*patch_size+v] += 2 / patch_size * alpha(u) * alpha(v) * np.cos((2*x+1)*u*np.pi/(2*patch_size)) * np.cos((2*y+1)*v*np.pi/(2*patch_size))
        P = P.astype(np.float32)

        filter = torch.zeros(patch_size**2, 1, patch_size, patch_size)
        for k in range(patch_size**2):
            filter[k, 0, k // patch_size, k % patch_size] = 1.0

        self.conv1 = nn.Conv2d(in_channels=1, 
                               out_channels=patch_size**2, 
                               kernel_size=patch_size, 
                               bias=False)
        self.conv1.weight = nn.Parameter(torch.from_numpy(P.T).view(patch_size**2, 1, patch_size, patch_size)) # based on P.T

        self.conv3 = nn.ConvTranspose2d(in_channels=patch_size**2, 
                                        out_channels=1, 
                                        kernel_size=patch_size, 
                                        bias=False)
        self.conv3.weight = nn.Parameter(filter) # filter
        self.conv3.requires_grad_(False)

        self.conv4 = nn.ConvTranspose2d(in_channels=1, 
                               out_channels=1, 
                               kernel_size=patch_size, 
                               bias=False)
        self.conv4.weight = nn.Parameter(torch.ones(1, 1, patch_size, patch_size)) # average
        self.conv4.requires_grad_(False)

        self.shrink = DifferentiableThreshold.apply

    def forward(self, y):

        y_conv1 = self.conv1(y) # N x p**2 x (H-p+1) x (W-p+1)
        shrink = self.shrink(y_conv1 / (3 * self.threshold)) # N x p**2 x (H-p+1) x (W-p+1)
        y_shrink = y_conv1 * shrink
        w = 1 / (1 + torch.sum(shrink, dim=1)) # N x (H-p+1) x (W-p+1)
        w = w.unsqueeze(1) # N x 1 x (H-p+1) x (W-p+1)

        y_conv2 = F.conv2d(y_shrink, torch.inverse(self.conv1.weight.view(self.patch_size**2, self.patch_size**2)).view(self.patch_size**2, self.patch_size**2, 1, 1)) # N x p**2 x (H-p+1) x (W-p+1)
        y_conv3 = self.conv3(y_conv2 * w) # N x 1 x H x W

        w_conv4 = self.conv4(w) # N x 1 x H x W
        out = y_conv3 / w_conv4 # N x 1 x H x W

        return out