import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

import cpt.utils
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from common.action_decoder import ActionDecoder, action_loss
from cpt.core_wrapper import core_wrapper
from PIL import Image
from torchvision import models as torchvision_models
from torchvision import transforms
from visual import utils
from visual.encoder_utils import build_visual_encoder
from visual.vision_transformer import DINOHead

from data import load_dataset
from bet.utils import build_bet

torchvision_archs = sorted(
    name
    for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__") and callable(torchvision_models.__dict__[name])
)


def get_arg_parser():

    parser = argparse.ArgumentParser("CPT-stage2", add_help=False)

    # Model parameters
    parser.add_argument(
        "--encoder_arch",
        default="vit_small",
        type=str,
        choices=["vit_tiny", "vit_small", "vit_base", "xcit", "deit_tiny", "deit_small"]
        + torchvision_archs
        + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""",
    )

    parser.add_argument(
        "--decoder_arch",
        default="resnet34",
        type=str,
        choices=["resnet34"] + torchvision_archs + torch.hub.list("facebookresearch/xcit:main"),
        help="""Name of architecture to train. For quick experiments with ViTs,
        we recommend using vit_tiny or vit_small.""",
    )

    parser.add_argument(
        "--patch_size",
        default=16,
        type=int,
        help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""",
    )


def train_cpt_stage2_bet(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    utils.wandb_init(args)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset, val_dataset = load_dataset(args, wrapper_cls="SeqVisDemoDataset", transform=transform)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=torch.utils.data.DistributedSampler(dataset, shuffle=True),
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        sampler=torch.utils.data.DistributedSampler(val_dataset, shuffle=False),
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # ============ building networks ... ============

    encoder, embed_dim = build_visual_encoder(args)

    # wrap with cpt core
    encoder = core_wrapper(encoder, embed_dim, args)

    print(f"Encoder embed_dim is {embed_dim}")

    action_decoder = build_bet(args, input_dim=embed_dim)
