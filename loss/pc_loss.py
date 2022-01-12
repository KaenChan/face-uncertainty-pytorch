#!/usr/bin/env python3
import logging

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F



class PC_Loss(nn.Module):
    def __init__(self):
        super(PC_Loss, self).__init__()
        self.loss_type = 'L2'
        print('PC_loss', self.loss_type)
        logging.info('PC_loss {}'.format(self.loss_type))

    def forward(self, mu_X, p_X, label):
        p_X = 1 - p_X  # change uncertainty to confidence prob
        mu_X = F.normalize(mu_X) # if mu_X was not normalized by l2
        mu_X = mu_X.detach()
        non_diag_mask = (1 - torch.eye(mu_X.size(0))).int()
        if label.device.type == 'cuda':
            non_diag_mask = non_diag_mask.cuda(0)
        if p_X.size(-1) != 0:
            p_X = p_X.mean(dim=-1, keepdim=True)
        cos_theta = F.linear(mu_X, mu_X)
        label_mask = (torch.eq(label[:, None], label[None, :])).int()
        pos_mask = (non_diag_mask * label_mask) > 0
        neg_mask = (non_diag_mask * (1-label_mask)) > 0

        p_fuse = torch.min(p_X, p_X.transpose(1, 0))

        if self.loss_type == 'L2':
            loss = (cos_theta - p_fuse) ** 2
        else:
            # L1
            loss = torch.abs(cos_theta - p_fuse)
        loss_pos = loss[pos_mask]
        self.loss_keep = 1.
        if self.loss_keep < 1:
            sorted, indices = torch.sort(loss_pos)
            loss_pos = sorted[:int(loss_pos.size()[0]*self.loss_keep)]
        loss_pos = loss_pos.mean()
        cosine_pos = cos_theta[pos_mask]
        cosine_neg = cos_theta[neg_mask]
        end_points = {}
        end_points['pconf'] = p_X
        end_points['cos'] = cosine_pos
        return loss_pos, end_points


if __name__ == "__main__":
    mls = PC_Loss()
    label = torch.Tensor([1, 2, 3, 2, 3, 3, 2])
    mu_data = np.array([[-1.7847768 , -1.0991699 ,  1.4248079 ],
                        [ 1.0405252 ,  0.35788524,  0.7338794 ],
                        [ 1.0620259 ,  2.1341069 , -1.0100055 ],
                        [-0.00963581,  0.39570177, -1.5577421 ],
                        [-1.064951  , -1.1261107 , -1.4181522 ],
                        [ 1.008275  , -0.84791195,  0.3006532 ],
                        [ 0.31099692, -0.32650718, -0.60247767]], dtype=np.float32)
    si_data = np.array([[0.28463233],
                        [0.10505871],
                        [0.3067418 ],
                        [0.024449  ],
                        [0.08328865],
                        [0.13096762],
                        [0.5405067 ]], dtype=np.float32)
    
    muX = torch.from_numpy(mu_data)
    siX = torch.from_numpy(si_data)
    print(muX.shape)
    diff = mls(muX, siX, label)
    print(diff)
    from utils.utils import display_watch_list
    print(display_watch_list(diff[1]))
