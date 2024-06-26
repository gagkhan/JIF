from torch import nn

from common.mlp import MLP


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
