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
        "--simloss",
        type=str,
        default="cross_entropy",
        choices=["cross_entropy", "l2", "l1"],
        help="""Type of loss used for the CPT training. We recommend using cross_entropy for most experiments.""",
    )

    parser.add_argument(
        "--center_update",
        type=utils.bool_flag,
        default=True,
        help="""Whether to apply centering to teacher predictions when using centering""",
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
        "--beta1",
        type=float,
        default=0.01,
        help="""Weight for the latent action regularization term.""",
    )

    parser.add_argument(
        "--beta2",
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

    # CPT Parameters
    parser.add_argument(
        "--core",
        type=str,
        default="lapo",
        choices=["ilpo", "lapo"],
        help="""The core method use to infer latent actions. The choices are ILPO and LAPO""",
    )

    parser.add_argument(
        "--latent_state_dim",
        type=int,
        default=16,
        help="""Dimensionality of the latent action i.e. output of the latent policy network""",
    )

    parser.add_argument(
        "--latent_action_dim",
        type=int,
        default=3,
        help="""Dimensionality of the latent action i.e. output of the latent policy network""",
    )
    parser.add_argument(
        "--policy_units",
        type=int,
        nargs="+",
        default=[512, 512],
        help="""Network size of Mlp used as the latent policy network""",
    )
    parser.add_argument(
        "--dynamics_units",
        type=int,
        nargs="+",
        default=[512, 512],
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

    # ============ building student and teacher networks ... ============
    student, _ = build_vitact_encoder(args)
    teacher, embed_dim = build_vitact_encoder(args)
    student_head = DINOHead(embed_dim, args.out_dim, args.use_bn_in_head)
    teacher_head = DINOHead(embed_dim, args.out_dim, args.use_bn_in_head)
    student: nn.Module = core_wrapper(student, embed_dim, args)
    action_decoder = ActionDecoder(
        latent_action_dim=args.latent_action_dim,
        units=args.action_decoder_units,
        action_shape=dataset.action_shape,
    )

    # move networks to gpu
    student = student.cuda()
    student_head = student_head.cuda()
    teacher = teacher.cuda()
    teacher_head = teacher_head.cuda()
    action_decoder = action_decoder.cuda()

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

    if utils.has_batchnorms(student_head):
        student_head = nn.SyncBatchNorm.convert_sync_batchnorm(student_head)
        teacher_head = nn.SyncBatchNorm.convert_sync_batchnorm(teacher_head)
        teacher_head = nn.parallel.DistributedDataParallel(teacher_head, device_ids=[args.gpu])
        teacher_head_without_ddp = teacher.module
    else:
        teacher_head_without_ddp = teacher_head

    student = nn.parallel.DistributedDataParallel(student, device_ids=[args.gpu])
    student_head = nn.parallel.DistributedDataParallel(student_head, device_ids=[args.gpu])
    # teacher and student start with the same weights
    teacher_without_ddp.load_state_dict(student.module.state_dict(), strict=False)
    teacher_head_without_ddp.load_state_dict(student_head.module.state_dict(), strict=False)
    # there is no backpropagation through the teacher, so no need for gradients
    for p in teacher.parameters():
        p.requires_grad = False
    for p in teacher_head.parameters():
        p.requires_grad = False
    print(f"Student and Teacher are built: they are both {args.encoder_arch} network.")

    # ============ preparing loss ... ============
    dino_loss = SimilarLoss(
        args.out_dim,
        args.warmup_teacher_temp,
        args.teacher_temp,
        args.warmup_teacher_temp_epochs,
        args.epochs,
        simloss=args.simloss,
        center_update=True,
    ).cuda()

    # ============ preparing optimizer ... ============
    params_groups = utils.get_params_groups(nn.ModuleList([student, student_head, action_decoder]))
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
    # momentum parameter is increased to 1. during training with a cosine schedule
    momentum_schedule = utils.cosine_scheduler(args.momentum_teacher, 1, args.epochs, len(data_loader))
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
            student_head,
            teacher,
            teacher_head,
            teacher_without_ddp,
            teacher_head_without_ddp,
            action_decoder,
            dino_loss,
            data_loader,
            optimizer,
            lr_schedule,
            wd_schedule,
            momentum_schedule,
            epoch,
            fp16_scaler,
            args,
        )

        val_stats = {}
        if epoch % 5 == 0:
            val_stats = validate(
                student,
                student_head,
                teacher,
                teacher_head,
                action_decoder,
                dino_loss,
                val_data_loader,
                epoch,
                fp16_scaler,
                args,
            )

        epoch_stats = {**train_stats, **val_stats}

        # ============ writing logs ... ============
        save_dict = {
            "student": student.state_dict(),
            "student_head": student_head.state_dict(),
            "teacher": teacher.state_dict(),
            "teacher_head": teacher_head.state_dict(),
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
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f"checkpoint{epoch:04}.pth"))
        log_stats = {**{f"{k}": v for k, v in epoch_stats.items()}, "epoch": epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            utils.wandb_log(epoch_stats, epoch=epoch)

            if epoch % 2 == 0:
                pass
                # cpt.utils.log_latent_umap(student, data_loader, epoch, args)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def train_one_epoch(
    student,
    student_head,
    teacher,
    teacher_head,
    teacher_without_ddp,
    teacher_head_without_ddp,
    action_decoder,
    dino_loss,
    data_loader,
    optimizer,
    lr_schedule,
    wd_schedule,
    momentum_schedule,
    epoch,
    fp16_scaler,
    args,
):

    # train mode
    for m in [student, student_head, teacher, teacher_head, action_decoder, dino_loss]:
        m.eval()

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
        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None):
            teacher_output = teacher_head(teacher(obs["next"]))
            latent_state, _, latent_actions, z_reg_loss, x_reg_loss = student(obs["curr"], obs["next"], obs["goal"])
            student_output = student_head(latent_state)
            dloss = dino_loss(student_output, teacher_output, epoch)
            z_reg_loss = torch.mean(z_reg_loss)
            x_reg_loss = torch.mean(x_reg_loss)
            predicted_action = action_decoder(latent_actions)
            aloss = action_loss(actions, predicted_action, amask)
            loss = dloss + args.alpha * aloss + args.beta1 * z_reg_loss + args.beta2 * x_reg_loss

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
                param_norms = utils.clip_gradients(action_decoder, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            optimizer.step()
        else:
            fp16_scaler.scale(loss).backward()
            if args.clip_grad:
                fp16_scaler.unscale_(optimizer)  # unscale the gradients of optimizer's assigned params in-place
                param_norms = utils.clip_gradients(student, args.clip_grad)
                param_norms = utils.clip_gradients(action_decoder, args.clip_grad)
            utils.cancel_gradients_last_layer(epoch, student, args.freeze_last_layer)
            fp16_scaler.step(optimizer)
            fp16_scaler.update()

        # EMA update for the teacher
        with torch.no_grad():
            m = momentum_schedule[it]  # momentum parameter

            student_backbone = student.module.encoder
            teacher_backbone = teacher_without_ddp

            for s, t in [(student_backbone, teacher_backbone), (student_head, teacher_head_without_ddp)]:
                for param_q, param_k in zip(s.parameters(), t.parameters()):
                    param_k.data.mul_(m).add_((1 - m) * param_q.detach().data)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(train_loss=loss.item())
        metric_logger.update(train_dloss=dloss.item())
        metric_logger.update(train_z_reg_loss=z_reg_loss.item())
        metric_logger.update(train_x_reg_loss=x_reg_loss.item())
        metric_logger.update(train_action_loss=aloss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validate(
    student,
    student_head,
    teacher,
    teacher_head,
    action_decoder,
    dino_loss,
    data_loader,
    epoch,
    fp16_scaler,
    args,
):

    # eval mode
    for m in [student, student_head, teacher, teacher_head, action_decoder, dino_loss]:
        m.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Validation: "
    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):

        obs = build_obs_dict(args, data_loader.dataset.keys, batch)
        actions = batch["actions"].cuda(non_blocking=True)
        amask = batch["amask"].cuda(non_blocking=True)

        # teacher and student forward passes + compute dino loss
        with torch.cuda.amp.autocast(fp16_scaler is not None) and torch.no_grad():
            teacher_output = teacher_head(teacher(obs["next"]))
            latent_state, _, latent_actions, z_reg_loss, x_reg_loss = student(obs["curr"], obs["next"], obs["goal"])
            student_output = student_head(latent_state)
            dloss = dino_loss(student_output, teacher_output, epoch)
            z_reg_loss = torch.mean(z_reg_loss)
            x_reg_loss = torch.mean(x_reg_loss)
            predicted_action = action_decoder(latent_actions)
            aloss = action_loss(actions, predicted_action, amask)
            loss = dloss + args.alpha * aloss + args.beta1 * z_reg_loss + args.beta2 * x_reg_loss

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        # logging
        torch.cuda.synchronize()
        metric_logger.update(val_loss=loss.item())
        metric_logger.update(val_dloss=dloss.item())
        metric_logger.update(val_z_reg_loss=z_reg_loss.item())
        metric_logger.update(val_x_reg_loss=x_reg_loss.item())
        metric_logger.update(val_action_loss=aloss.item())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


class SimilarLoss(nn.Module):

    def __init__(
        self,
        out_dim,
        warmup_teacher_temp,
        teacher_temp,
        warmup_teacher_temp_epochs,
        nepochs,
        student_temp=0.1,
        center_update=True,
        center_momentum=0.9,
        simloss="cross_entropy",
    ):
        super().__init__()
        self.student_temp = student_temp
        self.center_update = center_update
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))
        # we apply a warm up for the teacher temperature because
        # a too high temperature makes the training instable at the beginning
        self.teacher_temp_schedule = np.concatenate(
            (
                np.linspace(warmup_teacher_temp, teacher_temp, warmup_teacher_temp_epochs),
                np.ones(nepochs - warmup_teacher_temp_epochs) * teacher_temp,
            )
        )
        self.simloss = simloss
        assert simloss in ["cross_entropy", "l2", "l1"]

    def forward(self, student_output, teacher_output, epoch):
        """
        Cross-entropy between softmax outputs of the teacher and student networks.
        """
        student_out = student_output

        # teacher centering and sharpening
        teacher_out = teacher_output
        if self.center_update:
            teacher_out = teacher_output - self.center

        teacher_out = teacher_out.detach()

        if self.simloss == "cross_entropy":
            temp = self.teacher_temp_schedule[epoch]
            teacher_out = F.softmax(teacher_out / temp, dim=-1)
            student_out = student_output / self.student_temp
            loss = torch.sum(-teacher_out * F.log_softmax(student_out, dim=-1), dim=-1)
        elif self.simloss == "l2":
            loss = F.mse_loss(teacher_out, student_out)
        elif self.simloss == "l1":
            loss = F.l1_loss(teacher_out, student_out)
        total_loss = loss.mean()
        if self.center_update:
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CPT", parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_dino(args)
