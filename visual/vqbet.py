# Behavior Transformer

# Stage1: Vector quantization of action sequence

# curr_img, goal_img, actions
# actions to actions

# class ResidualVecQuant (multi layer quantization)


# Stage2: Transformer model to model action distribution

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, units=[64, 64], act_layer=nn.GELU):
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


class VectorQuantization(nn.Module):

    def __init__(self, input_dim, embed_dim=4, codebook_len=16):
        self.input_dim = input_dim
        self.embed_dim = embed_dim

        super().__init__()

        self.encoder = MLP(input_dim, embed_dim)
        self.decoder = MLP(embed_dim, input_dim)
        self.codebook = nn.Embedding(codebook_len, embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0, 0.5)

    def forward(self, x):
        z = self.encoder(x)
        idx = torch.argmin(torch.cdist(self.codebook.weight, z.unsqueeze(0)), dim=1)
        zq = self.codebook.weight[idx]
        zq = z + (zq - z).detach()
        xr = self.decoder(zq)

        recons_loss = torch.sum((x - xr) ** 2)

        # loss to bring the codebook closer to the encoder outputs
        vq_loss_term1 = torch.mean((z.detach() - zq) ** 2)
        # loss to bring the encoder outputs closer to the codebook
        vq_loss_term2 = torch.mean((z - zq.detach()) ** 2)
        vq_loss = vq_loss_term1 + vq_loss_term2

        return (
            recons_loss + vq_loss,
            recons_loss,
            vq_loss,
        )


def test_residual_vq_forward_pass():

    # 1) create model
    # 2) create a batch of random data
    # 3) forward pass and ..prints

    model = VectorQuantization(input_dim=5 * 3, embed_dim=4, codebook_len=16)
    x = torch.rand((32, 5 * 3))
    xr, loss = model(x)
    print(loss)


if __name__ == "__main__":
    test_residual_vq_forward_pass()
