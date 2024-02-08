import torch
from torch import nn
from torch.nn import functional as F


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, units=[64, 64], act_layer=nn.GELU):
        super(MLP, self).__init__()
        layers = []
        size_in = input_dim
        for size_out in units:
            layers.append(nn.Linear(size_in, size_out))
            layers.append(act_layer())
            size_in = size_out
        self.mlp = nn.Sequential(*layers)
        self.out = nn.Linear(size_in, output_dim)

    def forward(self, x):
        return self.out(self.mlp(x))


class Policy(nn.Module):

    def __init__(self, embed_dim, latent_action_dim, units=[64, 64]) -> None:
        self.embed_dim = embed_dim
        self.latent_action_dim = latent_action_dim
        self.units = units
        super(Policy, self).__init__()
        self.mlp = MLP(embed_dim * 2, latent_action_dim * 2, units)

    def forward(self, x):

        mu, log_std = self.mlp(x).chunk(2, dim=-1)
        # use rsample to get differentiable samples
        dist = torch.distributions.Normal(mu, log_std.exp())
        actions = dist.rsample()
        return actions


class Dynamics(nn.Module):

    def __init__(self, embed_dim, latent_action_dim, units=[64, 64]) -> None:
        self.embed_dim = embed_dim
        self.latent_action_dim = latent_action_dim
        self.units = units
        super(Dynamics, self).__init__()
        self.mlp = MLP(embed_dim + latent_action_dim, embed_dim, units)

    def forward(self, x):
        x = self.mlp(x)
        return x
