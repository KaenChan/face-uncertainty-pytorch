import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class UncertaintyConstraint(nn.Module):
    def __init__(self, loss_type='L1'):
        super(UncertaintyConstraint, self).__init__()
        self.loss_type = loss_type

    def forward(self, log_sigma_sq):
        sigma_sq = torch.exp(log_sigma_sq)
        if self.loss_type == 'L2':
            m0 = torch.mean(sigma_sq, dim=0, keepdim=True)
            m0 = torch.clamp(m0, min=1e-6)
            # m0 = torch.mean(sigma_sq)
            s_wd = (sigma_sq / m0 - 1) ** 2
            loss = torch.mean(s_wd)
        elif self.loss_type == 'L1':
            m0 = torch.mean(sigma_sq, dim=0, keepdim=True)
            m0 = torch.clamp(m0, min=1e-6)
            # m0 = torch.mean(sigma_sq)
            s_wd = torch.abs(sigma_sq / m0 - 1)
            loss = torch.mean(s_wd)
        return loss


if __name__ == "__main__":
    si_data = np.array([[-0.28463233, -2.5517333, 1.4781238],
                        [-0.10505871, -0.31454122, -0.29844758],
                        [-1.3067418, 0.48718405, 0.6779812],
                        [2.024449, -1.3925922, -1.6178994],
                        [-0.08328865, -0.396574, 1.0888542],
                        [0.13096762, -0.14382902, 0.2695235],
                        [0.5405067, -0.67946523, -0.8433032]], dtype=np.float32)
    siX = torch.from_numpy(si_data)
    uc = UncertaintyConstraint()
    v = uc(siX)
    print(v)
