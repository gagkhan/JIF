import os
from typing import Tuple

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
        assert(False)

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

            encoder = load_pretrained_weights(encoder, state_dict, key=pretrained_key)

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
