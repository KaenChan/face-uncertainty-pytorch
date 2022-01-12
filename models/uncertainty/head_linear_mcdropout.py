import torch
from torch.nn import Parameter
import numpy as np
from torch import nn


class LinearDropoutEmbedding(nn.Module):
    def __init__(self, in_channels, out_size):
        super(LinearDropoutEmbedding, self).__init__()
        self.fc = nn.Linear(in_channels, out_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x


class HeadLinearMCDropout(nn.Module):
    def __init__(self, in_channels, T=100):
        super(HeadLinearMCDropout, self).__init__()
        self.model = LinearDropoutEmbedding(in_channels, 512)
        self.model.train()
        self.T = T

    def forward(self, x):
        with torch.no_grad():
            # print(x)
            # print(x.shape)
            feats_list = [self.model(x) for _ in range(self.T)]
            # print(len(feats_list))
            feats = [l.cpu().numpy() for l in feats_list]
            # print('feats', np.array(feats).shape)
            feats = np.array(feats)
            feats = feats / np.linalg.norm(feats, axis=-1, keepdims=True)
            # print('feats', feats.shape)
            mu_mean = np.mean(feats, axis=0)  # 10000x256
            mu_mean = mu_mean / np.linalg.norm(mu_mean, axis=-1, keepdims=True)
            # print('mu_mean', feats.shape)

            mu_mean = np.array([mu_mean])
            score = np.sum(feats * mu_mean, axis=-1)
            # print('score', score)
            score_var = np.std(score, axis=0)
            score_var = score_var.reshape([-1, 1])
            # print(score_var)
            score_var = torch.from_numpy(score_var)
            return score_var

    def eval(self):
        def fix_bn(m):
            classname = m.__class__.__name__
            if classname.find('BatchNorm') != -1:
                m.eval()
        self.model.apply(fix_bn)  # fix batchnorm


if __name__ == "__main__":
    model = HeadLinearMCDropout(1024, 1)
    x = np.random.random((12, 1024)).astype(np.float32)
    x = torch.from_numpy(x)
    diff = model(x)
    print(diff)
