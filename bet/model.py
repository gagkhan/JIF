import torch
import torch.nn.functional as F
from bet.gpt import GPT
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, output_size, units):
        super().__init__()
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
        input_dim,
        context_len,
        num_actions,
        n_layer=4,
        n_head=2,
        n_embd=128,
        dropout=0.0,
        bias=True,
        causal=False,
    ):
        super().__init__()
        self.gpt = GPT(n_layer, n_head, n_embd, context_len, bias, dropout, causal)
        self.proj_in = nn.Linear(input_dim, n_embd)
        self.act_mlp = MLP(n_embd, num_actions, units=[64, 64])
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.n_embd = n_embd
        self.num_actions = num_actions

    def forward(self, x):
        B, T, *O = x.shape
        x = self.proj_in(x.view(B * T, *O))
        x = x.view(B, T, self.n_embd)
        x = self.gpt(x)
        p = self.act_mlp(x[:, -1])
        return p

    def loss(self, x, a):
        return self.cross_entropy_loss(self(x), a)

    @torch.no_grad()
    def act(self, x):
        p = F.softmax(self(x), dim=-1)
        actions = torch.multinomial(p, num_samples=1, replacement=True)
        return actions


def test_reshaping():

    n_embd = 16
    context_len = 4
    batch_size = 8

    x = torch.rand((batch_size, context_len, n_embd))
    B, T, *O = x.shape
    x = x.view(B * T, *O)
    assert x.shape[0] == B * T
    x = x.view(B, T, *O)
    assert x.shape[0] == B and x.shape[1] == T


def behavior_transformer(causal=False):

    input_dim = 16
    context_len = 4
    num_actions = 10
    n_layer = 4
    n_head = 2
    n_embd = 128
    dropout = 0.0
    bias = True
    batch_size = 8

    model = BeT(
        input_dim=input_dim,
        context_len=context_len,
        num_actions=num_actions,
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        dropout=dropout,
        bias=bias,
        causal=causal,
    )

    # test forward pass with BeT
    x = torch.rand((batch_size, context_len, input_dim))
    out = model(x)
    assert out.shape[0] == batch_size
    assert out.shape[1] == num_actions
    assert len(torch.nonzero(out, as_tuple=True)) > 0

    # create one hot action vectors
    actions = torch.zeros((batch_size, num_actions))
    actions[torch.arange(batch_size), torch.randint(0, num_actions, (batch_size,))] = 1

    print("loss: ", model.loss(x, actions))
    print("actions: ", model.act(x))


# network builders BeT for 'base' and 'large' sizes based on the sizes used in https://arxiv.org/pdf/2206.11251
# 'base' and 'large' are based on sizes used for push block and kitchen tasks respectively


def bet_base(input_dim, context_len, num_actions, causal):

    model = BeT(
        input_dim=input_dim,
        context_len=context_len,
        num_actions=num_actions,
        n_layer=4,
        n_head=4,
        n_embd=72,
        dropout=0.1,
        bias=False,
        causal=causal,
    )

    return model


def bet_large(input_dim, context_len, num_actions, causal):

    model = BeT(
        input_dim=input_dim,
        context_len=context_len,
        num_actions=num_actions,
        n_layer=6,
        n_head=6,
        n_embd=120,
        dropout=0.1,
        bias=False,
        causal=causal,
    )

    return model


def test_behavior_transformer_causal():
    behavior_transformer(causal=True)


def test_behavior_transformer():
    behavior_transformer(causal=False)


if __name__ == "__main__":

    test_behavior_transformer()
