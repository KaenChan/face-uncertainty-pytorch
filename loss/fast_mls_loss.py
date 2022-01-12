#!/usr/bin/env python3
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Fast_MLS_Loss(nn.Module):
    def __init__(self):
        super(Fast_MLS_Loss, self).__init__()

    def negMLS(self, mu_X, sigma_sq_X):
        cos_theta = F.linear(mu_X, mu_X)
        sigma_sq_fuse = sigma_sq_X + sigma_sq_X.transpose(0,1)
        diff = 2 * (1 - cos_theta) / (1e-10 + sigma_sq_fuse) + torch.log(1e-10 + sigma_sq_fuse)
        return diff, cos_theta

    def forward(self, mu_X, log_sigma_sq, label):
        mu_X = F.normalize(mu_X) # if mu_X was not normalized by l2
        non_diag_mask = (1 - torch.eye(mu_X.size(0))).int()
        if label.device.type == 'cuda':
            non_diag_mask = non_diag_mask.cuda(0)
        sigma_sq_X = torch.exp(log_sigma_sq)
        loss_mat, cosine = self.negMLS(mu_X, sigma_sq_X)
        label_mask = (torch.eq(label[:, None], label[None, :])).int()
        pos_mask = (non_diag_mask * label_mask) > 0
        neg_mask = (non_diag_mask * (1-label_mask)) > 0
        pos_loss = loss_mat[pos_mask]
        self.loss_keep = 1.
        if self.loss_keep < 1.:
            # t1 = np.percentile(pos_loss.cpu().detach().numpy(), 95)
            sorted, indices = torch.sort(pos_loss)
            pos_loss = sorted[:int(pos_loss.size()[0]*self.loss_keep)]
        pos_loss = pos_loss.mean()
        cosine_pos = cosine[pos_mask]
        cosine_neg = cosine[neg_mask]
        end_points = {}
        end_points['sigma'] = sigma_sq_X
        end_points['cos'] = cosine_pos
        return pos_loss, end_points


if __name__ == "__main__":
    mls = Fast_MLS_Loss()
    label = torch.Tensor([1, 2, 3, 2, 3, 3, 2])
    mu_data = np.array([[-1.7847768 , -1.0991699 ,  1.4248079 ],
                        [ 1.0405252 ,  0.35788524,  0.7338794 ],
                        [ 1.0620259 ,  2.1341069 , -1.0100055 ],
                        [-0.00963581,  0.39570177, -1.5577421 ],
                        [-1.064951  , -1.1261107 , -1.4181522 ],
                        [ 1.008275  , -0.84791195,  0.3006532 ],
                        [ 0.31099692, -0.32650718, -0.60247767]])
    si_data = np.array([[-0.28463233],
                        [-0.10505871],
                        [-1.3067418 ],
                        [ 2.024449  ],
                        [-0.08328865],
                        [ 0.13096762],
                        [ 0.5405067 ]])
    
    muX = torch.from_numpy(mu_data)
    siX = torch.from_numpy(si_data)
    print(muX.shape)
    diff = mls(muX, siX, label)
    print(diff)
