from typing import List

import torch
from cpt.core import FwdDyn, LatentActor, bottleneck_proj_mlp
from torch import nn


class LAPO(nn.Module):
    """
    Learning to Act without Actions https://arxiv.org/pdf/2312.10812 introduced Latent Action Policies (LAPO).

    This class implements the LAPO model, which is a neural network-based approach for learning to act without
    explicit action supervision. It consists of an encoder model to encode visual or multi-modal observations,
    a latent inverse dynamics model to predict latent actions given pairs of encoded observations, and a latent
    forward dynamics model to predict the next encoded observation given the current encoded observation and
    the predicted latent action.

    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        state_dim: int,
        action_dim: int,
        invdyn_units: List[int] = [64, 64],
        fwddyn_units: List[int] = [64, 64],
        action_cond=True,
        goal_cond=True,
        quantize_action=False,
        quantize_state=False,
    ) -> None:
        """
        Initialize the ILPOWrapper class.

        Args:
            encoder (nn.Module): The encoder model to encode visual or multi-modal observations.
            embed_dim (int): The dimension of the embedding of the output of the student transformer model.
            latent_action_dim (int): The dimension of the latent action.
            latent_invdyn_units (List[int], optional): The number of units in the latent policy network layers.
                Defaults to [64, 64].
            latent_fwddyn_units (List[int], optional): The number of units in the dynamics network layers.
                Defaults to [64, 64].
            latent_action_cond (bool, optional): A boolean indicating whether to condition the dynamics model
                on latent action. Defaults to True. When the dynamics model is not conditioned on latent action,
                it is instead conditioned on the goal embedding.
            quantize_latent_action (bool, optional): A boolean indicating whether to quantize the latent action.
                Defaults to True.
            quantize_latent_state (bool, optional): A boolean indicating whether to quantize the latent state.
                Defaults to False.

        """
        self.embed_dim = embed_dim
        self.action_dim = action_dim
        self.action_cond = action_cond
        self.goal_cond = goal_cond
        super().__init__()
        self.encoder = encoder
        self.proj_mlp = bottleneck_proj_mlp(3, embed_dim, state_dim, embed_dim, use_bn=False)
        self.invdyn = LatentActor(state_dim, action_dim, invdyn_units, quantize=quantize_action)
        if self.action_cond:
            self.fwddyn = FwdDyn(state_dim, action_dim, embed_dim, fwddyn_units, quantize=quantize_state)
        else:
            self.fwddyn = FwdDyn(state_dim, state_dim, embed_dim, fwddyn_units, quantize=quantize_state)

    def forward(self, o_curr, o_next, o_goal):
        x_curr = self.proj_mlp(self.encoder(o_curr))
        x_next = self.proj_mlp(self.encoder(o_next))

        if not self.goal_cond:
            x_next *= 0
        z_curr, _, z_reg_loss = self.invdyn(torch.cat([x_curr, x_next], dim=-1))
        if self.action_cond:
            x_next_pred, _, x_reg_loss = self.fwddyn(torch.cat([x_curr, z_curr], dim=-1))
        else:
            x_next_pred, _,  x_reg_loss = self.fwddyn(torch.cat([x_curr, x_next], dim=-1))
            zloss *= 0

        # NOTE: x_next is post-sampling or post-quantization, the pre-sampling or pre-quantized value stored in
        # x_next_pred is instead used.

        # return mean and variance of z

        return x_next_pred, x_curr, z_curr, z_reg_loss, x_reg_loss
