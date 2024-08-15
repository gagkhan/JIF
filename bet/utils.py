import bet.model as models
from bet.model import ActionDecoder
from torch import nn

def build_bet(args, input_img_dim) -> nn.Module:
    if args.bet_arch in models.__dict__.keys():
        return models.__dict__[args.bet_arch](input_img_dim, args.seq_len, args.causal)
    else:
        raise ValueError(f"{args.bet_arch} not defined")

def build_action_decoder(args, n_embd) -> nn.Module:
    return ActionDecoder(args.num_actions, n_embd, args.use_ee)