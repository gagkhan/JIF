import argparse
import datetime
import json
import math
import os
import sys
import time
import einops
import random
from itertools import islice
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
from data.sequential_multimodal import build_obs_dict as seq_build_obs_dict,load_dataset as seq_load_dataset
from data.multimodal import build_obs_dict, datakeys, load_dataset
from PIL import Image
from torchvision import models as torchvision_models
from torchvision import transforms
from visual import utils
from visual.vision_transformer import DINOHead
from visuotactile.utils import build_vitact_encoder


def get_args_parser():
    parser = argparse.ArgumentParser("CPT", add_help=False)

    # Model parameters
    parser.add_argument(
        "--encoder_arch",
        default="vitact_small",
        type=str,
        choices=["vitact_tiny", "vitact_small", "vitact_base"],
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
        "--use_bn_in_head",
        default=False,
        type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)",
    )

    # Training/Optimization parameters

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
        "--freeze_encoder",
        type=utils.bool_flag,
        default=False,
        help=""" Whether to freeze encoder (required to create the encoder with network builder)""",
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
        "--optimizer",
        default="adamw",
        type=str,
        choices=["adamw", "sgd", "lars"],
        help="""Type of optimizer. We recommend using adamw with ViTs.""",
    )
    parser.add_argument("--drop_path_rate", type=float, default=0.1, help="stochastic depth rate")

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
    parser.add_argument(
        "--train_split",
        default=0.9,
        type=float,
        help="split fraction of data for training, rest is used for validation",
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

    parser.add_argument(
        "--use_tactile",
        type=utils.bool_flag,
        default=True,
        help=""" Whether or not tactile data is used.""",
    )

    parser.add_argument(
        "--use_cam2",
        type=utils.bool_flag,
        default=True,
        help=""" Whether or not wrist view camera (cam2) is used.""",
    )

    parser.add_argument(
        "--use_cam3",
        type=utils.bool_flag,
        default=True,
        help=""" Whether or not wrist view camera (cam3) is used.""",
    )

    return parser


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
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    dataset, val_dataset = seq_load_dataset(args, datakeys(args), transform=transform)

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

    # ============ building student and teacher networks ... ============
    student, _ = build_vitact_encoder(args)
    # student: nn.Module = core_wrapper(student, embed_dim, args)

    # move networks to gpu
    student = student.cuda()

    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)

    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # there is no backpropagation through the teacher, so no need for gradients
    print(f"Student is built: it is a {args.encoder_arch} network.")


    # ============ preparing optimizer ... ============
    # params_groups = utils.get_params_groups(nn.ModuleList([student]))
    params_groups = utils.get_params_groups(student)
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
    # momentum parameter is increased to 1. during training with a cosine schedule
    # momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(data_loader))
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting CPT training !")

    best_val_loss = np.Infinity
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of CPT ... ============
        train_stats = train_one_epoch(
            student,
            data_loader,
            optimizer,
            lr_schedule,
            wd_schedule,
            epoch,
            fp16_scaler,
            args,
        )

        val_stats = {}
        if epoch % 5 == 0:
            val_stats = validate(
                student,
                val_data_loader,
                epoch,
                fp16_scaler,
                args,
            )

        epoch_stats = {**train_stats, **val_stats}

        # ============ writing logs ... ============
        save_dict = {
            "student": student.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
        }
        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, "checkpoint.pth"))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f"checkpoint{epoch:04}.pth"))
        if val_stats and val_stats["val_loss"] < best_val_loss:
            best_val_loss = val_stats["val_loss"]
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f"checkpoint_best.pth")) 
        log_stats = {**{f"{k}": v for k, v in epoch_stats.items()}, "epoch": epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            utils.wandb_log(epoch_stats, epoch=epoch)

            if epoch % 2 == 0:
                pass
                # cpt.utils.log_latent_umap([student, data_loader, epoch, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def train_one_epoch(
    student,
    data_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    epoch,
    fp16_scaler,
    args,
):

    # train mode
    student.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)
    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        random_curr = [[], [], []]
        obs = seq_build_obs_dict(args, data_loader.dataset.keys, batch)
        # print(obs["curr"])
        # print(len(obs["curr"]))
        # print(type(obs["curr"]))
        batch_size = args.batch_size_per_gpu
        for i in range(batch_size):
            rand_idx = random.randint(0, batch_size - 1)
            if rand_idx == i:
                rand_idx = (rand_idx + 1) % (batch_size - 1)
            for j in range(len(obs["curr"])):
                random_curr[j].append(obs["curr"][j][rand_idx])
            # random_batch.append(obs["curr"][i][rand_idx]) for i in len(obs["curr"])
        for j in range(len(obs["curr"])):
            random_curr[j] = torch.stack(random_curr[j])
        obs_j = obs["j"]
        obs_k = obs["k"]
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            # print(student(obs["curr"]))
            latent_curr = student(obs["curr"])
            latent_j = student(obs_j)
            latent_k = student(obs_k)
            latent_random_curr = student(random_curr)

            sim_i_j = -1 * F.mse_loss(latent_curr, latent_j)
            sim_i_k = -1 * F.mse_loss(latent_curr, latent_k)
            sim_i_rand = -1 * F.mse_loss(latent_curr, latent_random_curr)

            loss = - torch.log(torch.exp(sim_i_j) / (torch.exp(sim_i_j) + torch.exp(sim_i_k) + torch.exp(sim_i_rand)))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # student update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(train_loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validate(
    student,
    data_loader,
    epoch,
    fp16_scaler,
    args,
):

    # eval mode
    student.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Validation: "
    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        random_curr = [[], [], []]
        obs = seq_build_obs_dict(args, data_loader.dataset.keys, batch)
        # print(obs["curr"])
        # print(len(obs["curr"]))
        # print(type(obs["curr"]))
        batch_size = args.batch_size_per_gpu
        for i in range(batch_size):
            rand_idx = random.randint(0, batch_size - 1)
            if rand_idx == i:
                rand_idx = (rand_idx + 1) % (batch_size - 1)
            for j in range(len(obs["curr"])):
                random_curr[j].append(obs["curr"][j][rand_idx])
            # random_batch.append(obs["curr"][i][rand_idx]) for i in len(obs["curr"])
        for j in range(len(obs["curr"])):
            random_curr[j] = torch.stack(random_curr[j])
        obs_j = obs["j"]
        obs_k = obs["k"]

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None) and torch.no_grad():
            latent_curr = student(obs["curr"])
            latent_j = student(obs_j)
            latent_k = student(obs_k)
            latent_random_curr = student(random_curr)

            sim_i_j = -1 * F.mse_loss(latent_curr, latent_j)
            sim_i_k = -1 * F.mse_loss(latent_curr, latent_k)
            sim_i_rand = -1 * F.mse_loss(latent_curr, latent_random_curr)

            loss = - torch.log(torch.exp(sim_i_j) / (torch.exp(sim_i_j) + torch.exp(sim_i_k) + torch.exp(sim_i_rand)))

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(val_loss=loss.item())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

if __name__ == "__main__":
    parser = argparse.ArgumentParser("CPT", parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
