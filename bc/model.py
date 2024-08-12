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
            layers.append(nn.GELU())
            input_size = outsize
        layers.append(nn.Linear(input_size, output_size))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(x)


class DecoderMLP(nn.Module):
    def __init__(
        self,
        input_img_dim,
        input_ee_dim,
        action_dim,
        action_chunk_len,
        units,
        use_ee,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_chunk_len = action_chunk_len

        input_dim  = input_img_dim * 2 + (3 if use_ee else 0)
        output_dim = action_dim * action_chunk_len

        self.mlp = MLP(input_dim, output_dim, units)

    def forward(self, curr_img, goal_img, curr_ee=None): # use a list later
        """
        curr_img: (batch_size, input_img_dim)
        goal_img: (batch_size, input_img_dim)
        curr_ee:  (batch_size, input_ee_dim); optional
        """
        # Concatenate
        if curr_ee is None:
            x = torch.cat([curr_img, goal_img], dim=-1)
        else:
            x = torch.cat([curr_img, goal_img, curr_ee], dim=-1)
        
        # Forward
        p = self.mlp(x)

        # reshape
        p = p.view(-1, self.action_chunk_len, self.action_dim)

        return p


def mlp_large(input_img_dim, action_chunk_len, use_ee=False):

    model = DecoderMLP(
        input_img_dim=input_img_dim,
        input_ee_dim=3,
        action_dim=3,
        action_chunk_len=action_chunk_len,
        units=[512, 512],
        use_ee=use_ee,
    )

    return model


def mlp_base(input_img_dim, action_chunk_len, use_ee=False):

    model = DecoderMLP(
        input_img_dim=input_img_dim,
        input_ee_dim=3,
        action_dim=3,
        action_chunk_len=action_chunk_len,
        units=[64, 64],
        use_ee=use_ee
    )

    return model


def mlp_small (input_img_dim, action_chunk_len, use_ee=False):

    model = DecoderMLP(
        input_img_dim=input_img_dim,
        input_ee_dim=3,
        action_dim=3,
        action_chunk_len=action_chunk_len,
        units=[16, 16],
        use_ee=use_ee
    )

    return model