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

import ilpo
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import utils
import vision_transformer as vits
from data_utils import VisDemoDataset
from PIL import Image
from torchvision import datasets
from torchvision import models as torchvision_models
from torchvision import transforms
from vision_transformer import DINOHead
from data_aug import DataAugmentationCPT

torchvision_archs = sorted(
    name
    for name in torchvision_models.__dict__
    if name.islower() and not name.startswith("__") and callable(torchvision_models.__dict__[name])
)


def get_args_parser():
    parser = argparse.ArgumentParser("CPT", add_help=False)

    # Model parameters
    parser.add_argument(
        "--arch",
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
        "--out_dim",
        default=65536,
        type=int,
        help="""Dimensionality of
        the DINO head output. For complex and large datasets large values (like 65k) work well.""",
    )
    parser.add_argument(
        "--norm_last_layer",
        default=True,
        type=utils.bool_flag,
        help="""Whether or not to weight normalize the last layer of the DINO head.
        Not normalizing leads to better performance but can make the training unstable.
        In our experiments, we typically set this paramater to False with vit_small and True with vit_base.""",
    )
    parser.add_argument(
        "--momentum_teacher",
        default=0.996,
        type=float,
        help="""Base EMA
        parameter for teacher update. The value is increased to 1 during training with cosine schedule.
        We recommend setting a higher value with small batches: for example use 0.9995 with batch size of 256.""",
    )
    parser.add_argument(
        "--use_bn_in_head",
        default=False,
        type=utils.bool_flag,
        help="Whether to use batch normalizations in projection head (Default: False)",
    )

    # Temperature teacher parameters
    parser.add_argument(
        "--warmup_teacher_temp",
        default=0.04,
        type=float,
        help="""Initial value for the teacher temperature: 0.04 works well in most cases.
        Try decreasing it if the training loss does not decrease.""",
    )
    parser.add_argument(
        "--teacher_temp",
        default=0.04,
        type=float,
        help="""Final value (after linear warmup)
        of the teacher temperature. For most experiments, anything above 0.07 is unstable. We recommend
        starting with the default value of 0.04 and increase this slightly if needed.""",
    )
    parser.add_argument(
        "--warmup_teacher_temp_epochs",
        default=0,
        type=int,
        help="Number of warmup epochs for the teacher temperature (Default: 30).",
    )

    # Training/Optimization parameters
    parser.add_argument(
        "--measure",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "l2", "l1"],
        help="""Type of loss used for the CPT training. We recommend using cross_entropy for most experiments.""",
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

    # Multi-crop parameters
    parser.add_argument(
        "--global_crops_scale",
        type=float,
        nargs="+",
        default=(0.4, 1.0),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
        recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""",
    )
    parser.add_argument(
        "--local_crops_number",
        type=int,
        default=8,
        help="""Number of small
        local views to generate. Set this parameter to 0 to disable multi-crop training.
        When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """,
    )
    parser.add_argument(
        "--local_crops_scale",
        type=float,
        nargs="+",
        default=(0.05, 0.4),
        help="""Scale range of the cropped image before resizing, relatively to the origin image.
        Used for small local view cropping of multi-crop.""",
    )

    # ILPO parameters
    parser.add_argument("--latent_action_dim", type=int, default=128)
    parser.add_argument("--policy_units", type=int, nargs="+", default=[512, 512])
    parser.add_argument("--dynamics_units", type=int, nargs="+", default=[512, 512])
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
    parser.add_argument(
        "--output_dir", default=".", type=str, help="Path to save logs and checkpoints."
    )
    parser.add_argument(
        "--saveckp_freq", default=1000, type=int, help="Save checkpoint every x epochs."
    )
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument(
        "--num_workers", default=10, type=int, help="Number of data loading workers per GPU."
    )
    parser.add_argument(
        "--dist_url",
        default="env://",
        type=str,
        help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""",
    )
    parser.add_argument(
        "--local_rank", default=0, type=int, help="Please ignore and do not set this argument."
    )

    parser.add_argument(
        "--disable_wnb", default=False, type=utils.bool_flag, help="Disable wandb logging."
    )

    parser.add_argument(
        "--pretrained_weights",
        default="",
        type=str,
        help="Path to pretrained weights to load before training.",
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
    transform = DataAugmentationCPT(
        args.global_crops_scale,
        args.local_crops_scale,
        args.local_crops_number,
    )

    dataset = VisDemoDataset(
        data_root=args.data_path, transform=transform, skip_frames=args.skip_frames
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
    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building student and teacher networks ... ============
    # we changed the name DeiT-S for ViT-S to avoid confusions
    args.arch = args.arch.replace("deit", "vit")
    # if the network is a Vision Transformer (i.e. vit_tiny, vit_small, vit_base)
    if args.arch in vits.__dict__.keys():
        student = vits.__dict__[args.arch](
            patch_size=args.patch_size,
            drop_path_rate=args.drop_path_rate,  # stochastic depth
        )
        teacher = vits.__dict__[args.arch](patch_size=args.patch_size)
        embed_dim = student.embed_dim
    # if the network is a XCiT
    elif args.arch in torch.hub.list("facebookresearch/xcit:main"):
        student = torch.hub.load(
            "facebookresearch/xcit:main",
            args.arch,
            pretrained=False,
            drop_path_rate=args.drop_path_rate,
        )
        teacher = torch.hub.load("facebookresearch/xcit:main", args.arch, pretrained=False)
        embed_dim = student.embed_dim
    # otherwise, we check if the architecture is in torchvision models
    elif args.arch in torchvision_models.__dict__.keys():
        student = torchvision_models.__dict__[args.arch]()
        teacher = torchvision_models.__dict__[args.arch]()
        embed_dim = student.fc.weight.shape[1]
    else:
        print(f"Unknow architecture: {args.arch}")

    student_head = DINOHead(embed_dim, args.out_dim, args.use_bn_in_head)
    teacher_head = DINOHead(embed_dim, args.out_dim, args.use_bn_in_head)

    if args.pretrained_weights:
        state_dict = torch.load(args.pretrained_weights, map_location="cpu")

        def load_pretrained_weights(backbone, head, state_dict, key):
            backbone_state_dict = state_dict[key]
            # remove `module.` prefix
            backbone_state_dict = {
                k.replace("module.", ""): v for k, v in backbone_state_dict.items()
            }
            # remove `backbone.` prefix induced by multicrop wrapper
            backbone_state_dict = {
                k.replace("backbone.", ""): v for k, v in backbone_state_dict.items()
            }
            backbone.load_state_dict(backbone_state_dict, strict=False)
            head_state_dict = {}
            for k, v in backbone_state_dict.items():
                if "head" in k:
                    head_state_dict[k.replace("head.", "")] = v
            head.load_state_dict(head_state_dict)
            return backbone, head

        student, student_head = load_pretrained_weights(
            student, student_head, state_dict, key="student"
        )
        teacher, teacher_head = load_pretrained_weights(
            teacher, teacher_head, state_dict, key="teacher"
        )

    # multi-crop wrapper handles forward with inputs of different resolutions

    teacher = utils.MultiCropWrapper(
        teacher,
        teacher_head,
    )

    # ================== ILPO ==================
    # ILPO wrapper adds policy and dynamics networks
    student = ilpo.ILPOWrapper(
        utils.MultiCropWrapper(student),
        student_head,
        embed_dim,
        latent_action_dim=args.latent_action_dim,
        policy_units=args.policy_units,
        dynamics_units=args.dynamics_units,
    )

    action_decoder = ilpo.ActionDecoder(
        latent_action_dim=args.latent_action_dim,
        units=args.action_decoder_units,
        action_shape=dataset.action_shape,
    )

    # move networks to gpu
    student, teacher, action_decoder = student.cuda(), teacher.cuda(), action_decoder.cuda()

    # synchronize batch norms (if any)
    if utils.has_batchnorms(student):
        student = nn.SyncBatchNorm.convert_sync_batchnorm(student)
        teacher = nn.SyncBatchNorm.convert_sync_batchnorm(teacher)

        # we need DDP wrapper to have synchro batch norms working...
        teacher = nn.parallel.DistributedDataParallel(teacher, device_ids=[args.gpu])
        teacher_without_ddp = teacher.module
    else:
        # teacher_without_ddp and teacher are the same thing
        teacher_without_ddp = teacher
    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.arch} network.")

    # ============ preparing loss ... ============
    dino_loss = SimilarLoss(
        args.out_dim,
        args.local_crops_number + 2,  # total number of crops = 2 global crops + local_crops_number
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        measure=args.measure,
    ).cuda()

    kl_loss = KLLoss(args.local_crops_number + 2).cuda()

    # ============ preparing optimizer ... ============
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
    momentum_schedule = utils.cosine_scheduler(
        args.momentum_teacher, 1, args.epochs, len(data_loader)
    )
    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    utils.restart_from_checkpoint(
        os.path.join(args.output_dir, "checkpoint.pth"),
        run_variables=to_restore,
        student=student,
        teacher=teacher,
        action_decoder=action_decoder,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
        dino_loss=dino_loss,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting CPT training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)

        # ============ training one epoch of CPT ... ============
        train_stats = train_one_epoch(
            student,
            teacher,
            teacher_without_ddp,
            action_decoder,
            dino_loss,
            kl_loss,
            data_loader,
            optimizer,
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            epoch,
            fp16_scaler,
            args,
        )

        # ============ writing logs ... ============
        save_dict = {
            "student": student.state_dict(),
            "teacher": teacher.state_dict(),
            "action_decoder": action_decoder.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
            "dino_loss": dino_loss.state_dict(),
        }
        if fp16_scaler is not None:
            save_dict["fp16_scaler"] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(args.output_dir, "checkpoint.pth"))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(
                save_dict, os.path.join(args.output_dir, f"checkpoint{epoch:04}.pth")
            )
        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            utils.wandb_log(train_stats, epoch=epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def train_one_epoch(
    student,
    teacher,
    teacher_without_ddp,
    action_decoder,
    dino_loss,
    kl_loss,
    data_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    epoch,
    fp16_scaler,
    args,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)
    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):

        curr_images, next_images, goal_images, actions, amask = batch

        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu, use only one global view for the goal
        curr_images = [im.cuda(non_blocking=True) for im in curr_images]
        next_images = [im.cuda(non_blocking=True) for im in next_images]
        goal_images = [goal_images[0].cuda(non_blocking=True)] * len(curr_images)

        actions = actions.cuda(non_blocking=True)
        amask = amask.cuda(non_blocking=True)
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher(next_images)  # unlike DINO, all views pass through the teacher
            student_output, latent_actions, latent_mu, latent_logsigma = student(
                curr_images, goal_images
            )
            dloss = dino_loss(student_output, teacher_output, epoch)
            kloss = kl_loss(latent_mu, latent_logsigma)

            # compute action decoder loss
            # this needs to move to its own function or class
            latent_actions = latent_actions.chunk(args.local_crops_number + 2)
            aloss = 0
            for la in latent_actions:  # for each crop
                predicted_action = action_decoder(la)
                error = amask * (predicted_action - actions)
                # print("amask:", amask.shape)
                # print("error:", error.shape)
                sqerror = error * error
                # print("sqerror:", sqerror.shape)
                aloss += (sqerror).mean()
                # print("aloss:", aloss)
                # print(aloss.shape)
                # aloss += (torch.norm(amask * predicted_action - actions, dim=(1, 2)) ** 2).mean()
            aloss /= args.local_crops_number + 2
            loss = dloss + args.alpha * aloss + args.beta * kloss

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
                fp16_scaler.unscale_(
                    optimizer
                )  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter

            student_backbone = student.module.student.backbone
            teacher_backbone = teacher_without_ddp.backbone
            student_head = student.module.head
            teacher_head = teacher_without_ddp.head

            for s, t in [(student_backbone, teacher_backbone), (student_head, teacher_head)]:
                for param_q, param_k in zip(s.parameters(), t.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(dloss=dloss.item())
        metric_logger.update(kl_loss=kloss.item())
        metric_logger.update(action_loss=aloss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class SimilarLoss(nn.Module):
    def __init__(
        self,
        out_dim,
        ncrops,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_momentum=0.9,
        measure="cross_entropy",
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_momentum = center_momentum
        self.ncrops = ncrops
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )
        self.measure = measure
        assert measure in ["cross_entropy", "l2", "l1"]

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """

        student_out = student_output / self.student_temp
        student_out = student_out.chunk(self.ncrops)

        # teacher centering and sharpening
        temp = self.teacher_temp_schedule[epoch]
        teacher_out = teacher_output - self.center
        if self.measure == "cross_entropy":
            teacher_out = F.softmax(teacher_out / temp, dim=-1)

        teacher_out = teacher_out.detach().chunk(self.ncrops)
        # In DINO, we chunk into 2 outputs but here we chunk into ncrops because
        # all views pass through the teacher

        total_loss = 0
        n_loss_terms = 0
        for iq, q in enumerate(teacher_out):
            for v in range(len(student_out)):
                if self.measure == "cross_entropy":
                    loss = torch.sum(-q * F.log_softmax(student_out[v], dim=-1), dim=-1)
                elif self.measure == "l2":
                    loss = F.mse_loss(q, student_out[v])
                elif self.measure == "l1":
                    loss = F.l1_loss(q, student_out[v])
                total_loss += loss.mean()
                n_loss_terms += 1
        total_loss /= n_loss_terms
        self.update_center(teacher_output)
        return total_loss

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Update center used for teacher output.
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        dist.all_reduce(batch_center)
        batch_center = batch_center / (len(teacher_output) * dist.get_world_size())

        # ema update
        self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)


class KLLoss(nn.Module):
    def __init__(self, ncrops) -> None:
        super(KLLoss, self).__init__()
        self.ncrops = ncrops

    def forward(self, mu, logsigmas):
        mus = mu.chunk(self.ncrops)
        logsigmas = logsigmas.chunk(self.ncrops)

        kl_loss = 0
        for m, s in zip(mus, logsigmas):
            kl_loss += self._kl_loss(m, s)

        kl_loss /= self.ncrops

        return kl_loss

    def _kl_loss(self, s, m):
        return 0.5 * (s.exp().pow(2) + m.pow(2) - 2 * s - 1).mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CPT", parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
