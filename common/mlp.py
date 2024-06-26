from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, units=[64, 64], act_layer=nn.ReLU):
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
