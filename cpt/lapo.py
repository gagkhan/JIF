from typing import List

import torch
from torch import nn

from cpt.core import FwdDyn, LatentActor


class LAPO(nn.Module):
    """Wrapper around transformer encoder to add policy and dynamics networks"""

    def __init__(
        self,
        encoder: nn.Module,
        embed_dim: int,
        latent_action_dim: int,
        latent_invdyn_units: List[int] = [64, 64],
        latent_fwddyn_units: List[int] = [64, 64],
        latent_action_cond=True,
        quantize_latent_action=True,
        quantize_latent_state=False,
    ) -> None:
        """
        Initialize the ILPOWrapper class.

        Args:
            student: The student transformer model.
            head: The head model used to compute output later used to compute cross-entropy loss.
            embed_dim: The dimension of the embedding of the output of the student transformer model.
            latent_action_dim: The dimension of the latent action.
            policy_units: The number of units in the latent policy network layers. Defaults to [64, 64].
            dynamics_units: The number of units in the dynamics network layers. Defaults to [64, 64].
            latent_action_cond: A boolean indicating whether to condition the dynamics model on latent action.
                                Defaults to True. When dynamics model is not conditioned on latent action,
                                it is instead conditioned on the goal embedding.
            goal_cond: A boolean indicating whether to use goal cond i.e. when goal_cond=False the latent policy
                        will not be conditioned on the goal when latent_action_cond=True. Similarly, the forward
                        dynamics will not be conditioned on the goal when latent_action_cond=True
        """
        self.embed_dim = embed_dim
        self.latent_action_dim = latent_action_dim
        self.latent_action_cond = latent_action_cond
        super().__init__()
        self.encoder = encoder
        self.latent_invdyn = LatentActor(
            embed_dim,
            latent_action_dim,
            latent_invdyn_units,
        )

        if self.latent_action_cond:
            self.latent_fwddyn = FwdDyn(
                embed_dim,
                latent_action_dim,
                latent_fwddyn_units,
                quantize=quantize_latent_action,
            )
        else:
            self.latent_fwddyn = FwdDyn(
                embed_dim,
                embed_dim,
                latent_fwddyn_units,
                quantize=quantize_latent_action,
            )

        self.latent_fwddyn = FwdDyn(
            embed_dim,
            latent_action_dim,
            latent_invdyn_units,
            quantize=quantize_latent_state,
        )

    def forward(self, o_curr, o_next, o_goal):
        x_curr = self.encoder(o_curr)
        x_next = self.encoder(o_next)
        z_curr, _, zloss = self.latent_invdyn(torch.cat([x_curr, x_next], dim=-1))
        if self.latent_action_cond:
            x_next, x_next_pred, xloss = self.latent_fwddyn(
                torch.cat(
                    [x_curr, z_curr],
                    dim=-1,
                )
            )
        else:
            x_next, x_next_pred, xloss = self.latent_fwddyn(torch.cat([x_curr, x_next], dim=-1))
            zloss *= 0

        # x_next is after sampling or after quantization, since we are currently not regularizing the predictions, the quantization

        return x_next_pred, zloss, xloss
