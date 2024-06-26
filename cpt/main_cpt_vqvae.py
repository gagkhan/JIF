# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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

from common.action_decoder import ActionDecoder
from cpt.ilpo import ILPO
from data import VisDemoDataset
from visual import utils
from visual import vision_transformer as vits
from visual.decoder_utils import build_visual_decoder
from visual.encoder_utils import build_visual_encoder

# from visual.data_aug import DataAugmentationCPT
# from visual.vision_transformer import DINOHead

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
        "--lr",
        default=0.0005,
        type=float,
        help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""",
    )
    parser.add_argument(
        "--warmup_epochs",
        default=10,
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
        "--beta",
        type=float,
        default=0.01,
        help="""Weight for the latent action regularization term.""",
    )
    parser.add_argument(
        "--optimizer",
        default="adamw",
        type=str,
        choices=["adamw", "sgd", "lars"],
        help="""Type of optimizer. We recommend using adamw with ViTs.""",
    )
    parser.add_argument("--drop_path_rate", type=float, default=0.1, help="stochastic depth rate")

    # CPT parameters

    parser.add_argument(
        "--core",
        type=str,
        default="ilpo",
        choices=["ilpo", "lapo"],
        help="""The core method use to infer latent actions. The choices are ILPO and LAPO""",
    )

    parser.add_argument(
        "--latent_action_dim",
        type=int,
        default=128,
        help="""Dimensionality of the latent action i.e. output of the latent policy network""",
    )

    parser.add_argument(
        "--dynamics_units",
        type=int,
        nargs="+",
        default=[512, 512],
    )

    parser.add_argument(
        "--policy_units",
        type=int,
        nargs="+",
        default=[512, 512],
        help="""Network size of Mlp used as the latent policy network""",
    )

    parser.add_argument(
        "--action_decoder_units",
        type=int,
        nargs="+",
        default=[512, 512],
        help="""Network size of Mlp used as the action decoder network""",
    )
    parser.add_argument(
        "--latent_action_cond",
        type=utils.bool_flag,
        default=True,
        help="""A boolean indicating whether to condition the dynamics model on latent action. 
        Defaults to True.When dynamics model is not conditioned on latent action, it is instead
        conditioned on the goal embedding.""",
    )

    parser.add_argument(
        "--goal_cond",
        type=utils.bool_flag,
        default=True,
        help="""A boolean indicating whether to use goal cond i.e. when goal_cond=False the latent policy 
                will not be conditioned on the goal when latent_action_cond=True. Similarly, the forward 
                dynamics will not be conditioned on the goal when latent_action_cond=True""",
    )

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
        help="Path to pretrained weights to load before training.",
    )
    return parser


def action_loss(actions, actions_pred, mask):

    error = mask * (actions_pred - actions)
    sqerror = error * error
    aloss += (sqerror).mean()

    return aloss


def train_dino(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    utils.wandb_init(args)

    # ============ preparing data ... ============
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )
    dataset = VisDemoDataset(data_root=args.data_path, transform=transform, skip_frames=args.skip_frames)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building networks ... ============

    encoder, embed_dim = build_visual_encoder(args)

    print(f"Encoder embed_dim is {embed_dim}")
    decoder = build_visual_decoder(args)

    # ILPO wrapper adds policy and dynamics networks
    encoder = ILPO(
        encoder,
        embed_dim,
        latent_action_dim=args.latent_action_dim,
        latent_policy_units=args.policy_units,
        latent_fwddyn_units=args.dynamics_units,
        latent_action_cond=args.latent_action_cond,
    )

    action_decoder = ActionDecoder(
        latent_action_dim=args.latent_action_dim,
        units=args.action_decoder_units,
        action_shape=dataset.action_shape,
    )

    # move networks to gpu

    encoder, decoder, action_decoder = encoder.cuda(), decoder.cuda(), action_decoder.cuda()

    # synchronize batch norms (if any)
    if utils.has_batchnorms(encoder):
        encoder = nn.SyncBatchNorm.convert_sync_batchnorm(encoder)
    encoder = nn.parallel.DistributedDataParallel(encoder, device_ids=[args.gpu])

    print(f"Encoder is built: it is {args.encoder_arch} network.")
    print(f"Decoder is built: it is {args.decoder_arch} network.")

    # ============ preparing loss ... ============
    recon_loss = nn.MSELoss()
    # action_loss is already defined

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(nn.ModuleList([encoder, action_decoder]))
    # params_groups = utils.get_params_groups(student)
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
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        encoder=encoder,
        decoder=decoder,
        action_decoder=action_decoder,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        recon_loss=recon_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting CPT training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of CPT ... ============
        train_stats = train_one_epoch(
            encoder,
            decoder,
            action_decoder,
            recon_loss,
            data_loader,
            optimizer,
            lr_schedule,
            wd_schedule,
            epoch,
            fp16_scaler,
            args,
        )

        # ============ writing logs ... ============
        save_dict = {
            "encoder": encoder.state_dict(),  # BUG: This should be the unwrapped encoder.
            "decoder": decoder.state_dict(),
            "action_decoder": action_decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
            "recon_loss": recon_loss.state_dict(),
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
    decoder,
    action_decoder,
    recon_loss,
    data_loader,
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

        o_curr, o_next, o_goal, actions, amask = batch

        # o_curr.shape
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        o_curr = o_curr.cuda(non_blocking=True)
        o_next = o_next.cuda(non_blocking=True)

        if args.core == "ilpo":
            o_goal = o_goal.cuda(non_blocking=True)
        else:
            o_goal = None

        actions = actions.cuda(non_blocking=True)
        amask = amask.cuda(non_blocking=True)

        # pass through encoder and decoder and compute recons loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):

            # encoder outputs quantized latents and quantization loss
            x_next_pred, z_curr, z_reg_loss, x_reg_loss = encoder(o_curr, o_next, o_goal)

            # reconstruct
            o_next_pred = decoder(x_next_pred)
            actions_pred = action_decoder(z_curr)

            # accumulate losses
            rloss = recon_loss(o_next_pred, o_next)
            aloss = action_loss(actions_pred, actions, amask)
            loss = recon_loss + args.alpha * aloss + args.beta * z_reg_loss

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(encoder, args.clip_grad)
                param_norms = utils.clip_gradients(action_decoder, args.clip_grad)
                param_norms = utils.clip_gradients(decoder, args.clip_grad)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(encoder, args.clip_grad)
                param_norms = utils.clip_gradients(action_decoder, args.clip_grad)
                param_norms = utils.clip_gradients(decoder, args.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(dloss=rloss.item())
        metric_logger.update(kl_loss=z_reg_loss.item())
        metric_logger.update(action_loss=aloss.item())
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
    train_dino(args)
