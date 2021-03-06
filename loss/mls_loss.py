import logging

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class MLS_Loss(nn.Module):
    def __init__(self, loss_keep=0.99):
        super(MLS_Loss, self).__init__()
        self.loss_keep = 1.00
        logging.info('MLS_Loss keep {}'.format(self.loss_keep))

    def negMLS(self, mu_X, sigma_sq_X):
        # input.view(input.size(0), -1)
        D = mu_X.size(-1)
        mu_diff = mu_X.view(-1, 1, D) - mu_X.view(1, -1, D)
        sig_sum = sigma_sq_X.view(-1, 1, D) + sigma_sq_X.view(1, -1, D)
        diff = mu_diff*mu_diff / (1e-10 + sig_sum) + torch.log(sig_sum)  # BUG
        diff = diff.mean(dim=2, keepdim=False)
        return diff

    def forward(self, mu_X, log_sigma_sq, labels):
        mu_X = F.normalize(mu_X) # if mu_X was not normalized by l2
        mu_X = mu_X.detach()
        non_diag_mask = (1 - torch.eye(mu_X.size(0))).int()
        if labels.device.type == 'cuda':
            non_diag_mask = non_diag_mask.cuda(0)      
        sigma_sq_X = torch.exp(log_sigma_sq)
        loss_mat = self.negMLS(mu_X, sigma_sq_X)
        label_mask = (torch.eq(labels[:, None], labels[None, :])).int()
        pos_mask = (non_diag_mask * label_mask) > 0
        neg_mask = (non_diag_mask * (1-label_mask)) > 0
        pos_loss = loss_mat[pos_mask]
        if self.loss_keep < 1:
            # t1 = np.percentile(pos_loss.cpu().detach().numpy(), 95)
            sorted, indices = torch.sort(pos_loss)
            pos_loss = sorted[:int(pos_loss.size()[0]*self.loss_keep)]
        pos_loss = pos_loss.mean()
        # print(loss_mat[pos_mask])
        cosine = F.linear(mu_X, mu_X)
        cosine_pos = cosine[pos_mask]
        end_points = {}
        end_points['sigma'] = sigma_sq_X
        end_points['cos'] = cosine_pos
        return pos_loss, end_points


if __name__ == "__main__":
    mls = MLS_Loss()
    gty = torch.Tensor([1, 2, 3, 2, 3, 3, 2])
    mu_data = np.array([[-1.7847768 , -1.0991699 ,  1.4248079 ],
                        [ 1.0405252 ,  0.35788524,  0.7338794 ],
                        [ 1.0620259 ,  2.1341069 , -1.0100055 ],
                        [-0.00963581,  0.39570177, -1.5577421 ],
                        [-1.064951  , -1.1261107 , -1.4181522 ],
                        [ 1.008275  , -0.84791195,  0.3006532 ],
                        [ 0.31099692, -0.32650718, -0.60247767]], dtype=np.float32)
    si_data = np.array([[-0.28463233, -2.5517333 ,  1.4781238 ],
                        [-0.10505871, -0.31454122, -0.29844758],
                        [-1.3067418 ,  0.48718405,  0.6779812 ],
                        [ 2.024449  , -1.3925922 , -1.6178994 ],
                        [-0.08328865, -0.396574  ,  1.0888542 ],
                        [ 0.13096762, -0.14382902,  0.2695235 ],
                        [ 0.5405067 , -0.67946523, -0.8433032 ]], dtype=np.float32)
    
    muX = torch.from_numpy(mu_data)
    siX = torch.from_numpy(si_data)
    print(muX.shape)
    diff = mls(muX, siX, gty)
    print(diff)
