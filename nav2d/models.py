import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64, num_hidden=2):
        super(MLP, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.num_hidden = num_hidden

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(self.input_dim, self.hidden_dim))
        for _ in range(self.num_hidden):
            self.layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
        self.layers.append(nn.Linear(self.hidden_dim, self.output_dim))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = F.relu(layer(x))
        x = self.layers[-1](x)
        return x


class OIL(nn.Module):

    """ Observation only Imitation Learning (OIL) network"""

    def __init__(self, obs_dim, act_dim, hidden_dim=64, num_hidden=2):
        super(OIL, self).__init__()
        self.input_dim = obs_dim
        self.output_dim = act_dim
        self.forward_net = MLP(obs_dim+act_dim, obs_dim, hidden_dim, num_hidden)
        self.latent_action_net = MLP(obs_dim, act_dim, hidden_dim, num_hidden)
        self.action_net = MLP(act_dim, act_dim, hidden_dim, num_hidden)
        # NOTE: It is not necessary that the dimension of latent action is same as action. 
        #      You can use a different dimension for latent action and action. 

    def forward(self, x):
        # Compute latent action
        z = self.latent_action_net(x)
        x = torch.cat([x, z], dim=-1)

        # Compute next observation based on latent action and current observation
        y = self.forward_net(x)

        # Compute true action from latent action
        # NOTE: Detach latent action from the computation graph to avoid backpropagating 
        # through the action network. Currently we are interested only in understanding if 
        # the latent action can be used to the predict the the true action.
        a = self.action_net(z.detach())

        return y, a
