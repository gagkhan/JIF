from typing import List

import torch
from cpt.core import FwdDyn, LatentActor
from torch import nn


class ILPO(nn.Module):
    """Wrapper around visual or multi-modal encoder to add policy and dynamics networks .

    This class implements the ILPO (Imitating Latent Polcies from Observation) architecture.
    It serves as a wrapper around a encoder and adds policy and dynamics networks to the model.
    The ILPO model is described in the paper: https://arxiv.org/pdf/1805.07914

    """

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        action_dim: int,
        policy_units: List[int] = [64, 64],
        fwddyn_units: List[int] = [64, 64],
        action_cond=True,
        goal_cond=True,
        quantize_action=False,
        quantize_state=False,
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
        self.action_dim = action_dim
        self.action_cond = action_cond
        self.goal_cond = goal_cond
        super().__init__()
        self.encoder = encoder
        self.policy = LatentActor(embed_dim, action_dim, policy_units, quantize=quantize_action)

        if self.action_cond:
            self.fwddyn = FwdDyn(embed_dim, action_dim, fwddyn_units, quantize=quantize_state)
        else:
            self.fwddyn = FwdDyn(embed_dim, embed_dim, fwddyn_units, quantize=quantize_state)

    def forward(self, o_curr, o_next, o_goal):
        """
        Forward pass of the ILPO model.

        Args:
            o_curr: The current observation.
            o_next: The next observation.
            o_goal: The goal observation.

        Returns:
            x_next_pred: The predicted next observation.
            zloss: The loss for the latent policy.
            xloss: The loss for the forward dynamics model.
        """
        x_curr = self.encoder(o_curr)
        x_goal = self.encoder(o_goal)
        if not self.goal_cond:
            x_goal *= 0
        z_curr, mu, z_reg_loss = self.policy(torch.cat([x_curr, x_goal], dim=-1))
        if self.action_cond:
            _, x_next_pred, x_reg_loss = self.fwddyn(torch.cat([x_curr, z_curr], dim=-1))
        else:
            _, x_next_pred, x_reg_loss = self.fwddyn(torch.cat([x_curr, x_goal], dim=-1))
            z_reg_loss *= 0

        # NOTE: z_reg_loss and x_reg_loss stand for respective regularization losses.

        # NOTE: x_next is post-sampling or post-quantization, the pre-sampling or pre-quantized value stored in
        # x_next_pred is instead used.

        return x_next_pred, z_curr, z_reg_loss, x_reg_loss
