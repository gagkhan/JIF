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
        return actions, mu, log_std


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


class ILPOWrapper(nn.Module):
    """Wrapper around ViT model to add policy and dynamics networks"""

    def __init__(self, student, head, embed_dim, latent_action_dim, units=[64, 64]) -> None:
        self.embed_dim = embed_dim
        self.latent_action_dim = latent_action_dim
        self.units = units
        super(ILPOWrapper, self).__init__()
        self.student = student
        self.policy = Policy(embed_dim, latent_action_dim, units)
        self.dynamics = Dynamics(embed_dim, latent_action_dim, units)
        self.head = head

    def forward(self, ot, og):
        xt = self.student(ot)
        xg = self.student(og)
        x = torch.cat([xt, xg], dim=-1)
        zt, z_mu, z_sigma = self.policy(x)
        xtp1 = self.dynamics(torch.cat([xt, zt], dim=-1))
        return self.head(xtp1), zt, z_mu, z_sigma


class ActionDecoder(nn.Module):
    def __init__(self, latent_action_dim, units=[64, 64], dataset=None) -> None:
        self.latent_action_dim = latent_action_dim
        self.units = units
        super(ActionDecoder, self).__init__()
        self.action_shape = dataset[0][3].shape
        self.action_decoder_out_dim = len(dataset[0][3].reshape(-1))
        self.mlp = MLP(latent_action_dim, self.action_decoder_out_dim, units)

    def forward(self, x):
        out = self.mlp(x)
        out = out.view((-1, *self.action_shape))
        return out
