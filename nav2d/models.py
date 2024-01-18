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
        self.action_net = MLP(obs_dim, act_dim, hidden_dim, num_hidden)

    def forward(self, x):
        a = self.action_net(x)
        x = torch.cat([x, a], dim=-1)
        y = self.forward_net(x)
        return y, a
