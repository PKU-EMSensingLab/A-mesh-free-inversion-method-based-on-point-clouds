import numpy as np
import torch
import torch.nn as nn
from modules import *


class NICE(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.data_dim = cfg['PC_DIM'] + cfg['LATENT_DIM']
        self.hidden_dim = cfg['HIDDEN_DIM']
        self.num_layers = cfg['NUM_NET_LAYERS']
        self.z_dim = cfg['LATENT_DIM']
        self.num_coupling_layers = cfg['NUM_COUPLING_LAYERS']
        self.pc_dim = cfg['PC_DIM']
        self.latent_dim = cfg['LATENT_DIM']

        # alternating mask orientations for consecutive coupling layers
        masks = [self._get_mask(orientation=(i % 2 == 0)) for i in range(self.num_coupling_layers)]

        self.coupling_layers = nn.ModuleList([RNVPCouplingLayer(data_dim=self.data_dim,
                                    hidden_dim=self.hidden_dim,
                                    mask=masks[i], num_layers=self.num_layers)
                                for i in range(self.num_coupling_layers)])

        self.prior = GaussianDistribution(self.z_dim)

        self.encoder = nn.Sequential(nn.Linear(cfg['FIELD_DIM'], cfg['LATENT_DIM']))

        self.decoder = nn.Sequential(nn.Linear(cfg['LATENT_DIM'], cfg['FIELD_DIM']))

    def forward(self, x, pc, npoint, invert=False):
        if not invert:
            x = self.encoder(x)
            x = x.unsqueeze(2)
            x = x.repeat(1, 1, npoint)
            x = torch.cat([pc, x], dim=1)
            z, log_det_jacobian = self.f(x)
            log_likelihood = self.prior.log_prob(z[:, self.pc_dim:self.data_dim, :]) + log_det_jacobian
            return z, log_likelihood

        return self.f_inverse(x)

    def f(self, x):
        z = x
        log_det_jacobian = 0
        for i, coupling_layer in enumerate(self.coupling_layers):
            z, log_det_jacobian = coupling_layer(z, log_det_jacobian)
        return z, log_det_jacobian

    def f_inverse(self, z):
        x = z
        for i, coupling_layer in reversed(list(enumerate(self.coupling_layers))):
            x, _ = coupling_layer(x, 0, invert=True)
        return x

    def sample(self, pc, num_samples, npoint):
        z = self.prior.sample(num_samples, npoint)
        z = torch.cat([pc, z], dim=-2)
        z = self.f_inverse(z)
        x = z[:, self.pc_dim:self.pc_dim+self.latent_dim, :]
        x = x.mean(2)
        x = self.decoder(x)
        return x

    def _get_mask(self, orientation):
        mask = np.zeros((1, self.data_dim, 1))
        mask[:, ::2, :] = 1.
        if orientation:
            mask = 1. - mask     # flip mask orientation
        mask = torch.tensor(mask)
        mask = mask.cuda()

        return mask.float()
