import argparse
import datetime
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import datasets
from torchvision import models as torchvision_models
from torchvision import transforms

import visual.utils as utils
import visual.vision_transformer as vits
from bet.utils import build_bet
from bet.vq_actions import ActionVQVAE
from cpt import ilpo
from data import SeqVisDemoDataset
from visual.data_aug import DataAugmentationBC
from visual.encoder_utils import build_visual_encoder

torchvision_archs = sorted(
    name
    for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__") and callable(torchvision_models.__dict__[name])
)


def get_args_parser():
    parser = argparse.ArgumentParser("CPT", add_help=False)

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


def train_bc(args):

    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True
    
    utils.wandb_init(args)

    transform = DataAugmentationBC(args.naug)

    dataset = SeqVisDemoDataset(
        data_root=args.data_path,
        transform=transform,
        skip_frames=args.skip_frames,
        action_only=False,
        seq_len=args.context_len - 1,
        ac_len=args.action_chunk_len,
    )
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # TODO: action_scale to be fine tuned for the task
    action_quantizer = ActionVQVAE(
        action_dim=3, 
        action_chunk_size=args.action_chunk_len,
        encoder_units=[16,16,16],
        decoder_units=[16,16,16],
        embedding_dim=16,
        codebook_size=64,
        decay=0.9,
        use_vq_layer=True,
    )
    action_quantizer = action_quantizer.cuda()
    if args.pretrained_weights:
        action_quantizer.load_state_dict(torch.load( \
            "/ssd01/gagan/cpt_checkpoints/jul14_vqvae_tabletop_v0.1/checkpoint.pth" \
            )["action_quantizer"])

    print(f"Data loaded: there are {len(dataset)} demo frames.")

    # ============ building visual encoder network ... ============
    encoder, embed_dim = build_visual_encoder(args)

    encoder = utils.MultiCropWrapper(encoder)

    # ============ building policy network ... ============

    action_decoder = build_bet(args, input_dim=embed_dim)

    # move networks to gpu
    encoder, action_decoder = encoder.cuda(), action_decoder.cuda()
    # action_quantizer = action_quantizer.cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(nn.ModuleList([encoder, action_decoder]))
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    # ============ init schedulers ... ============
    '''
    lr_schedule = utils.cosine_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.0,  # linear scaling rule
        args.min_lr,
        args.epochs,
        len(data_loader),
        warmup_epochs=args.warmup_epochs,
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        len(data_loader),
    )
    '''
    lr_schedule = utils.linear_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        len(data_loader),
    )
    wd_schedule = utils.constant_scheduler(
        args.weight_decay,
        args.epochs,
        len(data_loader),
    )

    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=encoder,
        action_decoder=action_decoder,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()

    print("Starting BeT training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        # ============ training one epoch of BC ... ============
        train_stats = train_one_epoch(
            encoder,
            action_decoder,
            data_loader,
            action_quantizer,
            optimizer,
            lr_schedule,
            wd_schedule,
            epoch,
            fp16_scaler,
            args,
        )

        # ============ writing logs ... ============
        save_dict = {
            "encoder": encoder.state_dict(),
            "action_decoder": action_decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
        }
        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, "checkpoint.pth"))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f"checkpoint{epoch:04}.pth"))
        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            utils.wandb_log(train_stats, epoch=epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def train_one_epoch(
    encoder,
    action_decoder,
    data_loader,
    action_quantizer,
    optimizer,
    lr_schedule,
    wd_schedule,
    epoch,
    fp16_scaler,
    args,
):

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)
    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):

        img_seq, goal_images, actions, amask = batch

        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu, use only one global view for the goal
        curr_embd = []
        for img in img_seq:
            img = [im.cuda(non_blocking=True) for im in img]
            curr_embd.append(torch.vstack(encoder(img).chunk(args.naug + 1)))
        curr_embd = torch.stack(curr_embd, dim=1)
        goal_images = [im.cuda(non_blocking=True) for im in goal_images]
        goal_embd = torch.vstack(encoder(goal_images).chunk(args.naug + 1))

        actions = actions.repeat((args.naug + 1, 1, 1))
        amask = actions.repeat((args.naug + 1, 1, 1))

        # create one hot action vectors
        batch_size = curr_embd.shape[0]
        num_actions = action_decoder.num_actions
        onehot_actions = torch.zeros((batch_size, num_actions)).cuda()
        # onehot_actions[torch.arange(batch_size), torch.randint(0, num_actions, (batch_size,))] = 1
        _, idx = action_quantizer(actions)
        onehot_actions[torch.arange(batch_size), idx] = 1
        loss = action_decoder.loss(torch.cat([curr_embd, goal_embd.unsqueeze(1)], dim=1), onehot_actions)
        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # optimizer step
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(encoder, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, encoder, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(encoder, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, encoder, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(action_loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CPT", parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_bc(args)
