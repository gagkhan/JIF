import bet.model as bets
from torch import nn

def build_bet(args, input_dim) -> nn.Module:

    if args.bet_arch in bets.__dict__.keys():
        return bets.__dict__[args.bet_arch](input_dim, args.context_len, args.num_actions, args.causal)
    else:
        raise ValueError(f"{args.bet_arch} not defined")
