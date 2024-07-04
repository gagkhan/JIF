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
            action_dim=args.latent_action_dim,
            policy_units=args.policy_units,
            fwddyn_units=args.dynamics_units,
            action_cond=args.latent_action_cond,
        )
    elif args.core == "lapo":
        encoder = LAPO(
            encoder,
            embed_dim,
            action_dim=args.latent_action_dim,
            invdyn_units=args.policy_units,
            fwddyn_units=args.dynamics_units,
            action_cond=args.latent_action_cond,
        )
    else:
        raise ValueError(f"{args.core} is unknown")

    return encoder
