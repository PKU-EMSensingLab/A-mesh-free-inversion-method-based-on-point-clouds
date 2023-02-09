import torch
import torch.nn as nn
from torch.distributions import Distribution
import numpy as np
from torch.autograd import Variable
from math import log, pi
import copy

class RNVPCouplingLayer(nn.Module):

    def __init__(self, data_dim, hidden_dim, mask, num_layers):
        super().__init__()

        self.mask = mask

        modules = [nn.Conv1d(data_dim, hidden_dim, 1), nn.LeakyReLU(0.2, inplace=True)]
        for i in range(num_layers - 2):
            modules.append(nn.Conv1d(hidden_dim, hidden_dim, 1))
            modules.append(nn.LeakyReLU(0.2, inplace=True))
        modules.append(nn.Conv1d(hidden_dim, data_dim, 1))
        modules.append(nn.Tanh())

        self.s1 = nn.Sequential(*modules)
        self.t1 = nn.Sequential(*copy.deepcopy(modules))

        self.s2 = nn.Sequential(*copy.deepcopy(modules))
        self.t2 = nn.Sequential(*copy.deepcopy(modules))

    def forward(self, x, logdet, invert=False):
        if not invert:
            x1 = self.mask * x
            s, t = (1. - self.mask)*self.s1(x1), (1. - self.mask)*self.t1(x1)
            y = x1 + (1. - self.mask) * (x * torch.exp(s) + t)
            ldj = s.sum(dim=(1, 2))

            y1 = (1. - self.mask) * y
            s, t = self.mask*self.s2(y1), self.mask*self.t2(y1)
            y = y1 + self.mask * (y * torch.exp(s) + t)
            ldj = ldj + s.sum(dim=(1, 2))

            return y, logdet + ldj

        # Inverse additive coupling layer
        y1 = (1. - self.mask) * x
        s, t = self.mask*self.s2(y1), self.mask*self.t2(y1)
        y = y1 + self.mask * (x - t) * torch.exp(-s)
        ldj = -s.sum(dim=(1, 2))

        x1 = self.mask * y
        s, t = (1. - self.mask)*self.s1(x1), (1. - self.mask)*self.t1(x1)
        y = x1 + (1. - self.mask) * (y - t) * torch.exp(-s)
        ldj = ldj - s.sum(dim=(1, 2))
 
        return y, logdet + ldj


class GaussianDistribution(Distribution):
    def __init__(self, dim, scale=1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def log_prob(self, x):
        out = (-0.5*(x / self.scale)**2 - log(self.scale) -0.5 * log(2.0 * pi))
        out = out.sum(dim=(1,2))
        return out

    def sample(self, batch_size, npoint):
        out = np.random.normal(0, self.scale, (batch_size, self.dim, npoint))
        out = Variable(torch.Tensor(out)).cuda()
        return out