import os

import torch
import visual.vision_transformer as vits
from torchvision import models as torchvision_models


def build_visual_encoder(args):

    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.encoder_arch in vits.__dict__.keys():
        encoder = vits.__dict__[args.encoder_arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        embed_dim = encoder.embed_dim
    # if the network is a XCiT
    elif args.encoder_arch in torch.hub.list("facebookresearch/xcit:main"):
        encoder = torch.hub.load(
            "facebookresearch/xcit:main",
            args.encoder_arch,
            pretrained=False,
            drop_path_rate=args.drop_path_rate,
        )
        embed_dim = encoder.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.encoder_arch in torchvision_models.__dict__.keys():
        encoder = torchvision_models.__dict__[args.encoder_arch]()
        embed_dim = encoder.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.encoder_arch}")

    # Load pretrained weights
    if args.pretrained_weights:
        # Load local weights
        if os.path.isfile(args.pretrained_weights):
            state_dict = torch.load(args.pretrained_weights, map_location="cpu")

            def load_pretrained_weights(backbone, state_dict, key):
                backbone_state_dict = state_dict[key]
                # remove `module.` prefix
                backbone_state_dict = {k.replace("module.", ""): v for k, v in backbone_state_dict.items()}
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

    return encoder, embed_dim
