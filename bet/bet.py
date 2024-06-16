import torch
from gpt import GPT
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size, units):
        super(MLP, self).__init__()
        layers = []
        for outsize in units:
            layers.append(nn.Linear(input_size, outsize))
            layers.append(nn.ELU())
            input_size = outsize
        layers.append(nn.Linear(input_size, output_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class BeT(nn.Module):

    def __init__(
        self,
        obs_dim,
        hist_len,
        num_actions,
        n_layer=4,
        n_head=2,
        n_embd=128,
        dropout=0.0,
        bias=True,
        causal=False,
    ):
        self.gpt = GPT(n_layer, n_head, n_embd, hist_len, bias, dropout)
        self.act_mlp = MLP(n_embd, num_actions)
        self.cross_entropy_loss = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.gpt(x)
        p = self.act_mlp(x[0])

    def loss(self, x, a):
        return self.cross_entropy_loss(self(x), a)

    @torch.no_grad()
    def act(self, x):
        NT, choices = x.shape
        p = self(x)
        actions = torch.multinomial(p, choices)
        return actions
