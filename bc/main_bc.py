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
from PIL import Image

import visual.utils as utils
from bc.utils import build_mlp
from bc.args_parser import get_args_parser
from data import load_dataset
from visual.data_aug import DataAugmentationBC
from visual.encoder_utils import build_visual_encoder

from common.action_decoder import ActionDecoder


def train_bc(args):

    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    utils.wandb_init(args)

    transform = DataAugmentationBC(args.naug)

    # ============ Get dataloaders ... ============
    dataset, val_dataset = load_dataset(args, wrapper_cls="VisDemoDataset", transform=transform)
    sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
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

    print(f"Data loaded: there are {len(dataset)} images.")

    # ============ building visual encoder network ... ============
    encoder, embed_dim = build_visual_encoder(args)

    encoder = utils.MultiCropWrapper(encoder)

    encoder = encoder.cuda()

    # ============ building policy network ... ============
    latent_action_dim = 2 * embed_dim
    if args.use_ee: latent_action_dim += 3

    # action_decoder = ActionDecoder(
    #     latent_action_dim=latent_action_dim,
    #     units=[512, 512],
    #     action_shape=dataset.action_shape,
    # )
    action_decoder = build_mlp(args, embed_dim)

    # move networks to gpu
    action_decoder = action_decoder.cuda()

    # ============ preparing criterion ... ============
    criterion = get_loss
    
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
        student=encoder,
        action_decoder=action_decoder,
        optimizer=optimizer,
        fp16_scaler=fp16_scaler,
    )
    start_epoch = to_restore["epoch"]

    start_time = time.time()

    print("Starting BC training !")
    for epoch in range(start_epoch, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        # ============ training one epoch of BC ... ============
        
        train_stats = train_one_epoch(
            encoder,
            action_decoder,
            data_loader,
            criterion,
            optimizer,
            lr_schedule,
            wd_schedule,
            epoch,
            fp16_scaler,
            args,
        )
        val_stats = {}
        # if epoch % 5 == 0:
        #     val_stats = validate(
        #         encoder,
        #         action_decoder,
        #         val_data_loader,
        #         criterion,
        #         args,
        #     )
        
        epoch_stats = {**train_stats, **val_stats}

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
        log_stats = {**{f"train_{k}": v for k, v in epoch_stats.items()}, "epoch": epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            utils.wandb_log(epoch_stats, epoch=epoch)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))


def train_one_epoch(
    encoder,
    action_decoder,
    data_loader,
    criterion,
    optimizer,
    lr_schedule,
    wd_schedule,
    epoch,
    fp16_scaler,
    args,
):
    # prepare for training: put to train mode
    for m in [encoder, action_decoder]:
        m.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)
    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):

        if args.use_ee:
            curr_images, goal_images, curr_ee, actions, amask = batch
        else:
            curr_images, goal_images, actions, amask = batch
            curr_ee = None

        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu, use only one global view for the goal
        curr_images = [im.cuda(non_blocking=True) for im in curr_images]
        curr_embed = encoder(curr_images).chunk(args.naug + 1)

        goal_images = [goal_images[0].cuda(non_blocking=True)] * len(curr_images)
        goal_embed = encoder(goal_images).chunk(args.naug + 1)

        # move ee_sequences to gpu
        if args.use_ee:
            curr_ee = curr_ee.cuda(non_blocking=True)

        # move action labels to gpu
        true_actions = actions.cuda(non_blocking=True)
        amask = amask.cuda(non_blocking=True)

        # predict actions and loss
        loss = 0
        for curr, goal in zip(curr_embed, goal_embed):
            action_decoder_input = torch.cat([curr, goal, curr_ee], dim=-1)
            predicted_actions = action_decoder(curr, goal, curr_ee)
            loss += criterion(predicted_actions, true_actions, amask)
        loss = loss / (args.naug + 1)

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

        # logging metrics
        torch.cuda.synchronize()
        metric_logger.update(action_loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def validate(
    encoder,
    action_decoder,
    data_loader,
    criterion,
    args,
):
    # prepare for validation: put to eval mode
    for m in [encoder, action_decoder]:
        m.eval()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Validation: "
    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        with torch.no_grad():
            if args.use_ee:
                curr_images, goal_images, curr_ee, actions, amask = batch
            else:
                curr_images, goal_images, actions, amask = batch
                curr_ee = None

            # move images to gpu, use only one global view for the goal
            curr_images = [im.cuda(non_blocking=True) for im in curr_images]
            curr_embed = encoder(curr_images).chunk(args.naug + 1)

            goal_images = [goal_images[0].cuda(non_blocking=True)] * len(curr_images)
            goal_embed = encoder(goal_images).chunk(args.naug + 1)

            # move ee_sequences to gpu
            if args.use_ee:
                curr_ee = curr_ee.cuda(non_blocking=True)

            # move action labels to gpu
            true_actions = actions.cuda(non_blocking=True)
            amask = amask.cuda(non_blocking=True)

            # predict actions and loss
            loss = 0
            for curr, goal in zip(curr_embed, goal_embed):
                predicted_actions = action_decoder(curr, goal, curr_ee)
                loss += criterion(predicted_actions, true_actions, amask)
            loss = loss / (args.naug + 1)

            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), force=True)
                sys.exit(1)

            # logging metrics
            torch.cuda.synchronize()
            metric_logger.update(val_action_loss=loss.item())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def get_loss(output, target, mask):
    error = mask * (output - target)
    sqerror = error * error
    return sqerror.mean()


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CPT", parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_bc(args)
