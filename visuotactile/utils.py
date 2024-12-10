import os
from typing import Tuple, Callable

import torch
import visuotactile.visuo_tactile_transformer as vitact
from torch import nn
from torchvision import models as torchvision_models


def build_vitact_encoder(args) -> Tuple[nn.Module, int]:

    input_sizes = [(3, 224, 224)]
    patch_sizes = [args.patch_size]
    if args.use_tactile:
        input_sizes.append((2,))
        patch_sizes.append(1)
    if args.use_cam2:
        input_sizes.append((3, 224, 224))
        patch_sizes.append(args.patch_size)
    if args.use_cam3:
        input_sizes.append((3, 224, 224))
        patch_sizes.append(args.patch_size)

    # if the network is a Vision Transformer (i.e. vitact_tiny, vitact_small, vitact_base)
    if args.encoder_arch in vitact.__dict__.keys():
        encoder = vitact.__dict__[args.encoder_arch](
            input_sizes=input_sizes,
            patch_sizes=patch_sizes,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        embed_dim = encoder.embed_dim
    else:
        print(f"Unknow architecture: {args.encoder_arch}")

    # IMPORTANT!
    # replace all BatchNorm with GroupNorm to work with EMA
    # performance will tank if you forget to do this!
    encoder = replace_bn_with_gn(encoder)

    '''
    # Load pretrained weights
    if args.pretrained_weights:
        # Load local weights
        if os.path.isfile(args.pretrained_weights):
            state_dict = torch.load(args.pretrained_weights, map_location="cpu")

            def load_pretrained_weights(backbone, state_dict, key):
                backbone_state_dict = state_dict[key]
                # remove `module.` prefix
                backbone_state_dict = {k.replace("module.encoder.", ""): v for k, v in backbone_state_dict.items()}
                # remove `backbone.` prefix induced by multicrop wrapper
                backbone_state_dict = {k.replace("backbone.", ""): v for k, v in backbone_state_dict.items()}
                backbone.load_state_dict(backbone_state_dict, strict=False)
                return backbone

            encoder = load_pretrained_weights(encoder, state_dict, key="student")

        # Load online weights
        else:
            encoder = torchvision_models.__dict__[args.encoder_arch](weights=args.pretrained_weights)

        # Freeze pretrained weights
        if args.freeze_encoder:
            for p in encoder.parameters():
                p.requires_grad = False
            encoder.eval()

    # disable layers related to imagenet classification
    encoder.fc, encoder.head = nn.Identity(), nn.Identity()
    '''

    return encoder, embed_dim


def replace_submodules(
        root_module: nn.Module,
        predicate: Callable[[nn.Module], bool],
        func: Callable[[nn.Module], nn.Module]) -> nn.Module:
    """
    Replace all submodules selected by the predicate with
    the output of func.

    predicate: Return true if the module is to be replaced.
    func: Return new module to use.
    """
    if predicate(root_module):
        return func(root_module)

    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    for *parent, k in bn_list:
        parent_module = root_module
        if len(parent) > 0:
            parent_module = root_module.get_submodule('.'.join(parent))
        if isinstance(parent_module, nn.Sequential):
            src_module = parent_module[int(k)]
        else:
            src_module = getattr(parent_module, k)
        tgt_module = func(src_module)
        if isinstance(parent_module, nn.Sequential):
            parent_module[int(k)] = tgt_module
        else:
            setattr(parent_module, k, tgt_module)
    # verify that all modules are replaced
    bn_list = [k.split('.') for k, m
        in root_module.named_modules(remove_duplicate=True)
        if predicate(m)]
    assert len(bn_list) == 0
    return root_module

def replace_bn_with_gn(
    root_module: nn.Module,
    features_per_group: int=16) -> nn.Module:
    """
    Relace all BatchNorm layers with GroupNorm.
    """
    replace_submodules(
        root_module=root_module,
        predicate=lambda x: isinstance(x, nn.BatchNorm2d),
        func=lambda x: nn.GroupNorm(
            num_groups=x.num_features//features_per_group,
            num_channels=x.num_features)
    )
    return root_module
