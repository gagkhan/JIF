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

        mu, logsigma = self.mlp(x).chunk(2, dim=-1)
        # use rsample to get differentiable samples
        dist = torch.distributions.Normal(mu, logsigma.exp())
        actions = dist.rsample()
        return actions, mu, logsigma


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
    """Wrapper around transformer encoder to add policy and dynamics networks"""

    def __init__(
        self,
        student,
        head,
        embed_dim,
        latent_action_dim,
        policy_units=[64, 64],
        dynamics_units=[64, 64],
        latent_action_cond=False,
    ) -> None:
        """
        Initialize the ILPOWrapper class.

        Args:
            student: The student transformer model.
            head: The head model used to compute output later used to compute cross-entropy loss.
            embed_dim: The dimension of the embedding of the output of the student transformer model.
            latent_action_dim: The dimension of the latent action.
            policy_units: The number of units in the latent policy network layers. Defaults to [64, 64].
            dynamics_units: The number of units in the dynamics network layers. Defaults to [64, 64].
            latent_action_cond: A boolean indicating whether to condition the dynamics model on latent action. Defaults to True.
                                When dynamics model is not conditioned on latent action, it is instead conditioned on the goal embedding.
        """
        self.embed_dim = embed_dim
        self.latent_action_dim = latent_action_dim
        self.latent_action_cond = latent_action_cond
        super(ILPOWrapper, self).__init__()
        self.student = student
        self.latent_policy = Policy(embed_dim, latent_action_dim, policy_units)
        if self.latent_action_cond:
            self.latent_dynamics = Dynamics(embed_dim, latent_action_dim, dynamics_units)
        else:
            self.latent_dynamics = Dynamics(embed_dim, embed_dim, dynamics_units)
        self.head = head

    def forward(self, ot, og):
        xt = self.student(ot)
        xg = self.student(og)
        x = torch.cat([xt, xg], dim=-1)
        zt, z_mu, z_logsigma = self.latent_policy(x)
        if self.latent_action_cond is False:
            xtp1 = self.latent_dynamics(torch.cat([xt, xg], dim=-1))
        else:
            xtp1 = self.latent_dynamics(torch.cat([xt, zt], dim=-1))
        return self.head(xtp1), zt, z_mu, z_logsigma


class ActionDecoder(nn.Module):
    def __init__(self, latent_action_dim, units=[64, 64], action_shape=None) -> None:
        self.latent_action_dim = latent_action_dim
        self.units = units
        super(ActionDecoder, self).__init__()
        self.action_shape = action_shape
        self.action_decoder_out_dim = action_shape[0] * action_shape[1]
        self.mlp = MLP(latent_action_dim, self.action_decoder_out_dim, units)

    def forward(self, x):
        out = self.mlp(x)
        out = out.view((-1, *self.action_shape))
        return out
