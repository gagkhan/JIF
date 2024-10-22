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
from common.action_decoder import ActionDecoder, cartesian_loss, quaternion_loss
from cpt.core import LatentActor
from cpt.core_wrapper import core_wrapper
from PIL import Image
from torchvision import models as torchvision_models
from torchvision import transforms
from visual import utils
from visuotactile.utils import build_vitact_encoder

from data.multimodal import load_dataset, datakeys, build_obs_dict

torchvision_archs = sorted(
    name
    for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__") and callable(torchvision_models.__dict__[name])
)

from cpt.core import MLP


def get_args_parser():

    parser = argparse.ArgumentParser("CPT-stage2", add_help=False)

    # Encoder and student args
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
        "--latent_action_dim",
        type=int,
        default=16,
        help="""Dimensionality of the latent action i.e. output of the latent policy network""",
    )

    # Others
    parser.add_argument(
        "--use_fp16",
        type=utils.bool_flag,
        default=False,
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
        default=128,
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

    parser.add_argument(
        "--action_decoder_units",
        type=int,
        nargs="+",
        default=[32],
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
        "--use_cam2",
        type=utils.bool_flag,
        default=True,
        help=""" Whether or not wrist view camera (cam3) is used.""",
    )

    parser.add_argument(
        "--use_cam3",
        type=utils.bool_flag,
        default=True,
        help=""" Whether or not wrist view camera (cam3) is used.""",
    )

    parser.add_argument(
        "--use_tactile",
        type=utils.bool_flag,
        default=True,
        help=""" Whether or not tactile data is used.""",
    )

    parser.add_argument(
        "--use_ee",
        type=utils.bool_flag,
        default=False,
        help=""" Whether or not ee data is used.""",
    )

    return parser


class LatentPolicy(nn.Module):
    """Policy network to be pretrained for BC"""

    def __init__(self, input_dim: int, latent_action_dim: int, units=[64, 64]) -> None:

        self.input_dim = input_dim
        self.units = units
        super().__init__()
        self.mlp = MLP(input_dim, 2 * latent_action_dim, units)

    def forward(self, x):

        assert x.shape[1] == self.input_dim, "The input shape is not correct, some problem configuring input_dim"
        z = self.mlp(x)
        z_mu, z_logsigma = z.chunk(2, dim=-1)
        return z_mu, z_logsigma


def train(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    utils.wandb_init(args)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    dataset, val_dataset = load_dataset(args, datakeys(args), transform=transform)
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
    # From now on, we only use dataset for its data attributes
    if len(dataset) == 0:
        dataset = val_dataset
    args.shapes_dict = dataset.shapes_dict

    # ============ building networks ... ============

    args.pretrained_weights = ""
    encoder, embed_dim = build_vitact_encoder(args)
    encoder = encoder.cuda()

    student_input_dim = embed_dim + embed_dim * args.goal_cond
    student = LatentPolicy(input_dim=student_input_dim, latent_action_dim=args.latent_action_dim, units=[64, 64])
    student = student.cuda()

    action_decoder_input_dim = args.latent_action_dim + dataset.shapes_dict["ee_pose"] * args.use_ee
    action_decoder = ActionDecoder(
        latent_action_dim=action_decoder_input_dim,
        units=args.action_decoder_units,
        action_shape=dataset.action_shape,
    )
    action_decoder = action_decoder.cuda()
    
    # # Load state dict
    # state_dict = torch.load("/ssd01/gagan/cpt_checkpoints/10_15_bc_try_v0.1/checkpoint_best.pth", weights_only=False)
    # encoder.load_state_dict(state_dict["encoder"], strict=True)
    # student.load_state_dict(state_dict["student"], strict=True)
    # action_decoder.load_state_dict(state_dict["action_decoder"], strict=True)

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(nn.ModuleList([encoder, student, action_decoder]))
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
    lr_schedule = utils.linear_scheduler(
        args.lr * (args.batch_size_per_gpu * utils.get_world_size()) / 256.0,  # linear scaling rule
        args.min_lr,
        args.epochs,
        len(data_loader),
    )
    wd_schedule = utils.cosine_scheduler(
        args.weight_decay,
        args.weight_decay_end,
        args.epochs,
        len(data_loader),
    )

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        action_decoder=action_decoder,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()

    print("Starting CPT-Stage2 (BC) training !")

    best_val_loss = np.Infinity
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        # ============ training one epoch of BC ... ============
        train_stats = {}
        if len(data_loader) != 0:
            train_stats = train_one_epoch(
                encoder,
                student,
                action_decoder,
                data_loader,
                optimizer,
                lr_schedule,
                wd_schedule,
                epoch,
                fp16_scaler,
                args,
            )
        val_stats = {}
        if len(val_data_loader) != 0:
            val_stats = validate(
                encoder,
                student,
                action_decoder,
                val_data_loader,
                epoch,
                args,
            )

        epoch_stats = {**train_stats, **val_stats}

        # ============ writing logs ... ============
        save_dict = {
            "encoder": encoder.state_dict(),
            "student": student.state_dict(),
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
        if val_stats["val_loss"] < best_val_loss:
            best_val_loss = val_stats["val_loss"]
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f"checkpoint_best.pth")) 
        log_stats = {**{f"{k}": v for k, v in epoch_stats.items()}, "epoch": epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            utils.wandb_log(epoch_stats, epoch=epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def train_one_epoch(
    encoder,
    student,
    action_decoder,
    data_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    epoch,
    fp16_scaler,
    args,
):

    # train mode
    for m in [encoder, student, action_decoder]:
        m.train()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)
    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):

        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        obs = build_obs_dict(args, data_loader.dataset.keys, batch)
        actions = batch["actions"].cuda(non_blocking=True)
        amask = batch["amask"].cuda(non_blocking=True)

        x_curr = encoder(obs["curr"])
        if args.goal_cond:
            x_goal = encoder(obs["goal"])
        else:
            x_goal = torch.empty(x_curr.shape[0], 0).cuda()
        z_student, z_logsigma = student(torch.cat([x_curr, x_goal], dim=-1))
        actions_pred = action_decoder(torch.cat([z_student, obs["ee"].cuda()], dim=-1)) if obs["ee"] is not None else action_decoder(z_student)

        aloss = cartesian_loss(actions_pred[:,:,0:3], actions[:,:,0:3], amask[:,:,0:3])
        gloss = cartesian_loss(actions_pred[:,:,[7]], actions[:,:,[7]], amask[:,:,[7]])
        qloss = quaternion_loss(actions_pred[:,:,3:7], actions[:,:,3:7], amask[:,:,3:7])
        loss = args.alpha * (1.50 * aloss + 1.0 * gloss + qloss)
        metric_logger.update(train_gloss=gloss.item())

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # network update
        optimizer.zero_grad()
        param_norms = None
        if fp16_scaler is None:
            loss.backward()
            if args.clip_grad:
                param_norms = utils.clip_gradients(encoder, args.clip_grad)
                param_norms = utils.clip_gradients(action_decoder, args.clip_grad)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(encoder, args.clip_grad)
                param_norms = utils.clip_gradients(action_decoder, args.clip_grad)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(train_loss=loss.item())
        metric_logger.update(train_aloss=aloss.item())
        metric_logger.update(train_qloss=qloss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats


def validate(
    encoder,
    student,
    action_decoder,
    data_loader,
    epoch,
    args,
):

    for m in [encoder, student, action_decoder]:
        m.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)
    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):

        with torch.no_grad():

            obs = build_obs_dict(args, data_loader.dataset.keys, batch)
            actions = batch["actions"].cuda(non_blocking=True)
            amask = batch["amask"].cuda(non_blocking=True)

            x_curr = encoder(obs["curr"])
            if args.goal_cond:
                x_goal = encoder(obs["goal"])
            else:
                x_goal = torch.empty(x_curr.shape[0], 0).cuda()
            z_student, z_logsigma = student(torch.cat([x_curr, x_goal], dim=-1))
            actions_pred = action_decoder(torch.cat([z_student, obs["ee"].cuda()], dim=-1)) if obs["ee"] is not None else action_decoder(z_student)

            aloss = cartesian_loss(actions_pred[:,:,0:3], actions[:,:,0:3], amask[:,:,0:3])
            gloss = cartesian_loss(actions_pred[:,:,[7]], actions[:,:,[7]], amask[:,:,[7]])
            qloss = quaternion_loss(actions_pred[:,:,3:7], actions[:,:,3:7], amask[:,:,3:7])
            loss = args.alpha * (1.50 * aloss + 1.0 * gloss + qloss)
            metric_logger.update(val_gloss=gloss.item())

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(val_loss=loss.item())
        metric_logger.update(val_aloss=aloss.item())
        metric_logger.update(val_qloss=qloss.item())

        # np.set_printoptions(precision=3, suppress=True)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CPT", parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)
