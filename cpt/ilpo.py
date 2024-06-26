from typing import List

import torch
from torch import nn

from cpt.core import FwdDyn, LatentActor


class ILPO(nn.Module):
    """Wrapper around transformer encoder to add policy and dynamics networks"""

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        latent_action_dim: int,
        latent_policy_units: List[int] = [64, 64],
        latent_fwddyn_units: List[int] = [64, 64],
        latent_action_cond=True,
        goal_cond=True,
        quantize_latent_action=True,
        quantize_latent_state=False,
    ) -> None:
        """
        Initialize the ILPO class.

        Args:
            encoder: The encoder module used in the ILPO model.
            embed_dim: The dimension of the embedding of the input to the encoder.
            latent_action_dim: The dimension of the latent action.
            latent_policy_units: A list of integers specifying the number of units in the latent policy network layers.
                Defaults to [64, 64].
            latent_fwddyn_units: A list of integers specifying the number of units in the forward dynamics network layers.
                Defaults to [64, 64].
            latent_action_cond: A boolean indicating whether to condition the forward dynamics model on the latent action.
                Defaults to True.
            goal_cond: A boolean indicating whether to condition the latent policy and forward dynamics models on the
                goal embedding. Defaults to True.
            quantize_latent_action: A boolean indicating whether to quantize the latent action. Defaults to True.
            quantize_latent_state: A boolean indicating whether to quantize the latent state. Defaults to False.
        """
        self.embed_dim = embed_dim
        self.latent_action_dim = latent_action_dim
        self.latent_action_cond = latent_action_cond
        self.goal_cond = goal_cond
        super().__init__()
        self.encoder = encoder
        self.latent_policy = LatentActor(embed_dim, latent_action_dim, latent_policy_units)

        if self.latent_action_cond:
            self.latent_fwddyn = FwdDyn(
                embed_dim, latent_action_dim, latent_fwddyn_units, quantize=quantize_latent_action
            )
        else:
            self.latent_fwddyn = FwdDyn(embed_dim, embed_dim, latent_fwddyn_units, quantize=quantize_latent_action)

        self.latent_fwddyn = FwdDyn(embed_dim, latent_action_dim, latent_policy_units, quantize=quantize_latent_state)

    def forward(self, o_curr, o_next, o_goal):
        x_curr = self.encoder(o_curr)
        x_goal = self.encoder(o_goal)
        if not self.goal_cond:
            x_goal *= 0
        z_curr, mu, zloss = self.latent_policy(torch.cat([x_curr, x_goal], dim=-1))
        if self.latent_action_cond:
            _, x_next_pred, xloss = self.latent_fwddyn(torch.cat([x_curr, z_curr], dim=-1))
        else:
            _, x_next_pred, xloss = self.latent_fwddyn(torch.cat([x_curr, x_goal], dim=-1))
            zloss *= 0

        # NOTE: x_next is post-sampling or post-quantization, the pre-sampling or pre-quantized value stored in
        # x_next_pred is instead used.
        return x_next_pred, zloss, xloss
