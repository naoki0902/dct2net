import numpy as np
import torch
import torch.nn as nn


def alpha(u):
    if u == 0:
        return 1 / np.sqrt(2)
    else:
        return 1


class DCTDenoiser(nn.Module):

    def __init__(self, patch_size, threshold):

        super(DCTDenoiser, self).__init__()

        self.patch_size = patch_size
        self.threshold = threshold

        P = np.zeros((patch_size**2, patch_size**2))
        for x in range(patch_size):
            for y in range(patch_size):
                for u in range(patch_size):
                    for v in range(patch_size):
                        P[x*patch_size+y, u*patch_size+v] += 2 / patch_size * alpha(u) * alpha(v) * np.cos((2*x+1)*u*np.pi/(2*patch_size)) * np.cos((2*y+1)*v*np.pi/(2*patch_size))
        P = P.astype(np.float32) # これによってself.Pinv@inputができるようになった
        self.P = nn.Parameter(torch.from_numpy(P))
        self.Pinv = nn.Parameter(torch.from_numpy(P.T))
        self.P.requires_grad_(False)
        self.Pinv.requires_grad_(False)

        self.unfold = nn.Unfold(kernel_size=patch_size)
        self.shrink = nn.Hardshrink(threshold)

    def forward(self, y):

        _, _, height, width = y.shape # N x 1 x H x W
        self.fold = nn.Fold(output_size=(height, width), kernel_size=self.patch_size)

        y = self.unfold(y) # N x p^2 x (H-p+1)*(W-p+1)
        y = self.Pinv @ y # N x p^2 x (H-p+1)*(W-p+1)
        y = self.shrink(y) # N x p^2 x (H-p+1)*(W-p+1)
        w = 1 / (1 + torch.count_nonzero(y, dim=1)) # N x (H-p+1)*(W-p+1)
        y = self.P @ y # N x p^2 x (H-p+1)*(W-p+1)
        y *= w # N x p^2 x (H-p+1)*(W-p+1)
        y = self.fold(y) # N x 1 x H x W
        
        W = self.fold(w.unsqueeze(1).repeat(1, self.patch_size**2, 1)) # N x 1 x H x W
        output = y / W # N x 1 x H x W

        return output