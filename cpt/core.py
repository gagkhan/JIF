from typing import List

import torch
from torch import nn
from torch.nn import functional as F

from common.mlp import MLP
from vector_quantize_pytorch import VectorQuantize


class KLLoss(nn.Module):
    def __init__(self) -> None:
        super(KLLoss, self).__init__()

    def forward(self, mu, logsigma):
        kl_loss = self._kl_loss(logsigma, mu)
        return kl_loss

    def _kl_loss(self, s, m):
        return 0.5 * (s.exp().pow(2) + m.pow(2) - 2 * s - 1).mean()


class LatentInferBase(nn.Module):
    """Latent inference base class that can be used towards"""

    def __init__(self, input_dim: int, output_dim: int, units=[64, 64], quantize=True) -> None:

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.units = units
        self.quantize = quantize
        super(LatentInferBase, self).__init__()

        if not self.quantize:
            self.mlp = MLP(input_dim, output_dim, units)
            self.kl_loss = KLLoss()
        else:
            self.quantizer = VectorQuantize(
                codebook_dim=output_dim,
                codebook_size=64,
                decay=0.8,
                commitment_weight=1.0,
            )
            self.mlp = MLP(input_dim, output_dim, units)

    def forward(self, x):

        if not self.quantize:

            mu, logsigma = self.mlp(x).chunk(2, dim=-1)
            # use rsample to get differentiable samples
            dist = torch.distributions.Normal(mu, logsigma.exp())

            latents = dist.rsample()

            loss = self.kl_loss(mu, logsigma)

        else:
            mu = self.mlp(x)
            latents, loss, _ = self.quantizer(z)

        return latents, mu, loss


class LatentActor(LatentInferBase):

    def __init__(self, embed_dim, latent_action_dim, units=[64, 64], quantize=True):

        self.embed_dim = embed_dim
        self.latent_action_dim = latent_action_dim

        super(LatentActor, self).__init__(
            input_dim=2 * embed_dim,
            output_dim=latent_action_dim,
            units=units,
            quantize=quantize,
        )


class FwdDyn(LatentInferBase):

    def __init__(self, embed_dim, latent_action_dim, units=[64, 64], quantize=True) -> None:

        self.embed_dim = embed_dim
        self.latent_action_dim = latent_action_dim

        super(FwdDyn, self).__init__(
            input_dim=embed_dim + latent_action_dim,
            output_dim=embed_dim,
            units=units,
            quantize=quantize,
        )
