import bc.model as bc_models
from torch import nn

def build_mlp(args, input_img_dim) -> nn.Module:
    if args.bc_arch in bc_models.__dict__.keys():
        return bc_models.__dict__[args.bc_arch](input_img_dim, args.action_chunk_len, args.use_ee)
    else:
        raise ValueError(f"{args.bet_arch} not defined")
