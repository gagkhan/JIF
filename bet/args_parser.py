import argparse

import torch
from torchvision import models as torchvision_models

import visual.utils as utils


def get_args_parser():
    parser = argparse.ArgumentParser("CPT", add_help=False)

    # Model parameters
    torchvision_archs = sorted(
        name
        for name in torchvision_models.__dict__
        if name.islower() and not name.startswith("__") and callable(torchvision_models.__dict__[name])
    )

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
        "--patch_size",
        default=16,
        type=int,
        help="""Size in pixels
        of input square patches - default 16 (for 16x16 patches). Using smaller
        values leads to better performance but requires more memory. Applies only
        for ViTs (vit_tiny, vit_small and vit_base). If <16, we recommend disabling
        mixed precision training (--use_fp16 false) to avoid unstabilities.""",
    )

    parser.add_argument(
        "--use_fp16",
        type=utils.bool_flag,
        default=True,
        help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.04,
        help="""Initial value of the
        weight decay. With ViT, a smaller value at the beginning of training works well.""",
    )
    parser.add_argument(
        "--weight_decay_end",
        type=float,
        default=0.4,
        help="""Final value of the
        weight decay. We use a cosine schedule for WD and using a larger decay by
        the end of training improves performance for ViTs.""",
    )
    parser.add_argument(
        "--clip_grad",
        type=float,
        default=3.0,
        help="""Maximal parameter
        gradient norm if using gradient clipping. Clipping with norm .3 ~ 1.0 can
        help optimization for larger ViT architectures. 0 for disabling.""",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        default=64,
        type=int,
        help="Per-GPU batch-size : number of distinct images loaded on one GPU.",
    )
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs of training.")
    parser.add_argument(
        "--freeze_last_layer",
        default=1,
        type=int,
        help="""Number of epochs
        during which we keep the output layer fixed. Typically doing so during
        the first epoch helps training. Try increasing this value if the loss does not decrease.""",
    )
    parser.add_argument(
        "--lr",
        default=0.0005,
        type=float,
        help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""",
    )
    parser.add_argument(
        "--warmup_epochs",
        default=1,
        type=int,
        help="Number of epochs for the linear learning-rate warm up.",
    )
    parser.add_argument(
        "--min_lr",
        type=float,
        default=1e-6,
        help="""Target LR at the
        end of optimization. We use a cosine LR schedule with linear warmup.""",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=10,
        help="""Weight for the action decoder predictions.""",
    )

    parser.add_argument(
        "--optimizer",
        default="adamw",
        type=str,
        choices=["adamw", "sgd", "lars"],
        help="""Type of optimizer. We recommend using adamw with ViTs.""",
    )
    parser.add_argument("--drop_path_rate", type=float, default=0.1, help="stochastic depth rate")

    # Data-augmentation parameters
    parser.add_argument(
        "--naug",
        type=int,
        # nargs="+",
        default=0,
        help="""Number of noisy data augmentations to include alongside the original image""",
    )

    parser.add_argument("--action_decoder_units", type=int, nargs="+", default=[512, 512])

    # Misc
    parser.add_argument(
        "--data_path",
        default="/path/to/imagenet/train/",
        type=str,
        help="Please specify path to the ImageNet training data.",
    )
    parser.add_argument(
        "--skip_frames",
        default=5,
        type=int,
        help="Number of frames to skip when loading the dataset.",
    )
    parser.add_argument("--output_dir", default=".", type=str, help="Path to save logs and checkpoints.")
    parser.add_argument("--saveckp_freq", default=1000, type=int, help="Save checkpoint every x epochs.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument("--num_workers", default=10, type=int, help="Number of data loading workers per GPU.")
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""",
    )
    parser.add_argument("--local_rank", default=0, type=int, help="Please ignore and do not set this argument.")

    parser.add_argument("--disable_wnb", default=False, type=utils.bool_flag, help="Disable wandb logging.")

    parser.add_argument(
        "--pretrained_weights",
        default="",
        type=str,
        help="Path to pretrained weights or name of online weights to load before training.",
    )

    parser.add_argument(
        "--freeze_encoder",
        action="store_true",
        help="Freezes the encoder weights during training",
    )
    parser.add_argument(
        "--use_ee",
        action="store_true",
        help="Whether the action decode input includes ee position",
    )

    parser.add_argument(
        "--bet_arch",
        choices=["bet_small", "bet_base", "bet_large"],
        help="The architecture of the behavior transformer to choose from",
    )

    # add arguments for BeT like context_len, num_actions etc,.

    parser.add_argument(
        "--context_len",
        type=int,
        default=6,
        help="Context length of the behavior transformer. Note that the context includes goal making the history length, context length minus one.",
    )

    parser.add_argument(
        "--num_actions",
        type=int,
        default=13,
        help="Number of discrete actions in the action space of the behavior transformer",
    )

    parser.add_argument(
        "--action_chunk_len",
        default=6,
        type=int,
        help="Number of true actions for action chunking",
    )

    parser.add_argument(
        "--causal",
        default=False,
        action="store_true",
        help="Number of true actions for action chunking",
    )

    # Add arguments for ActionVQVAE

    parser.add_argument(
        "--action_quantizer_encoder_units",
        type=int,
        nargs="+",
        default=[16,16,16],
    )

    parser.add_argument(
        "--action_quantizer_decoder_units",
        type=int,
        nargs="+",
        default=[16,16,16],
    )

    parser.add_argument(
        "--action_quantizer_embedding_dim",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--action_quantizer_codebook_size",
        type=int,
        default=64,
    )

    parser.add_argument(
        "--action_quantizer_decay",
        type=float,
        default=0.9,
    )

    parser.add_argument(
        "--action_quantizer_use_vq_layer",
        action="store_true",
        help="Whether to use vq layer in action_quantizer; when not set, action_quantizer becomes an autoencoder",
    )

    return parser