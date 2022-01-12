#!/usr/bin/env python3
import torch
import torch.nn.functional as F
from torch.nn import Linear, Conv2d, BatchNorm1d, BatchNorm2d, PReLU, ReLU, Sigmoid, Dropout, MaxPool2d, \
    AdaptiveAvgPool2d, Sequential, Module
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter


class Flatten(Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class HeadLinearProb(nn.Module):
    ''' Evaluate the log(sigma^2) '''
    
    def __init__(self, in_feat=512, out_size=1):
        super(HeadLinearProb, self).__init__()
        assert out_size == 1
        self.layers = Sequential(Flatten(),
                                 Linear(in_feat, 1),
                                 Sigmoid()
                                 )

    def forward(self, x):
        x = self.layers(x)
        return x


if __name__ == "__main__":
    unh = HeadLinearProb(in_feat=3)
    mu_data = np.array([[-1.7847768 , -1.0991699 ,  1.4248079 ],
                        [ 1.0405252 ,  0.35788524,  0.7338794 ],
                        [ 1.0620259 ,  2.1341069 , -1.0100055 ],
                        [-0.00963581,  0.39570177, -1.5577421 ],
                        [-1.064951  , -1.1261107 , -1.4181522 ],
                        [ 1.008275  , -0.84791195,  0.3006532 ],
                        [ 0.31099692, -0.32650718, -0.60247767]])
    muX = torch.from_numpy(mu_data).float()
    sigma_sq = unh(muX)
    print(sigma_sq)
