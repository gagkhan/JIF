import torch
import torch.nn.functional as F
from bet.gpt import GPT
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
        super().__init__()
        self.gpt = GPT(n_layer, n_head, n_embd, hist_len, bias, dropout)
        self.act_mlp = MLP(n_embd, num_actions, units=[64, 64])
        self.obs_enc = MLP(obs_dim, n_embd, units=[64, 64])
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.n_embd = n_embd

    def forward(self, x):
        B, T, *O = x.shape
        x = self.obs_enc(x.view(B * T, *O))
        x = x.view(B, T, self.n_embd)
        x = self.gpt(x)
        p = F.softmax(self.act_mlp(x[:, -1]), dim=-1)
        return p

    def loss(self, x, a):
        return self.cross_entropy_loss(self(x), a)

    @torch.no_grad()
    def act(self, x):
        p = self(x)
        actions = torch.multinomial(p, num_samples=1, replacement=True)
        return actions


def test_reshaping():

    obs_dim = 10
    hist_len = 4
    batch_size = 8

    x = torch.rand((batch_size, hist_len, obs_dim))
    B, T, *O = x.shape
    x = x.view(B * T, *O)
    assert x.shape[0] == B * T
    x = x.view(B, T, *O)
    assert x.shape[0] == B and x.shape[1] == T


def behavior_transformer(causal=False):

    obs_dim = 10
    hist_len = 4
    num_actions = 10
    n_layer = 4
    n_head = 2
    n_embd = 128
    dropout = 0.0
    bias = True
    batch_size = 8

    model = BeT(
        obs_dim=obs_dim,
        hist_len=hist_len,
        num_actions=num_actions,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias,
        causal=causal,
    )

    # test forward pass with BeT
    x = torch.rand((batch_size, hist_len, obs_dim))
    out = model(x)
    assert out.shape[0] == batch_size
    assert out.shape[1] == num_actions
    assert len(torch.nonzero(out, as_tuple=True)) > 0

    # create one hot action vectors
    actions = torch.zeros((batch_size, num_actions))
    actions[torch.arange(batch_size), torch.randint(0, num_actions, (batch_size,))] = 1

    print("loss: ", model.loss(x, actions))
    print("actions: ", model.act(x))


def test_behavior_transformer_causal():
    behavior_transformer(True)


def test_behavior_transformer():
    behavior_transformer(False)


if __name__ == "__main__":

    test_behavior_transformer()
