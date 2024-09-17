from cpt.ilpo import ILPO
from cpt.lapo import LAPO
from torch import nn


def core_wrapper(
    encoder,
    embed_dim,
    args,
) -> nn.Module:

    # ILPO wrapper adds policy and dynamics networks
    if args.core == "ilpo":
        encoder = ILPO(
            encoder,
            embed_dim,
            state_dim=args.latent_state_dim,
            action_dim=args.latent_action_dim,
            policy_units=args.policy_units,
            fwddyn_units=args.dynamics_units,
            action_cond=args.latent_action_cond,
            goal_cond=args.goal_cond,
            quantize_action=args.quantize_action,
            quantize_state=args.quantize_state,
        )
    elif args.core == "lapo":
        encoder = LAPO(
            encoder,
            embed_dim,
            state_dim=args.latent_state_dim,
            action_dim=args.latent_action_dim,
            invdyn_units=args.policy_units,
            fwddyn_units=args.dynamics_units,
            action_cond=args.latent_action_cond,
            goal_cond=args.goal_cond,
            quantize_action=args.quantize_action,
            quantize_state=args.quantize_state,
        )
    else:
        raise ValueError(f"{args.core} is unknown")

    return encoder
