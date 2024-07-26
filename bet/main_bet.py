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
from torchvision.ops import sigmoid_focal_loss

import visual.utils as utils
import visual.vision_transformer as vits
from bet.utils import build_bet
from bet.vq_actions import ActionVQVAE
from bet.args_parser import get_args_parser
from cpt import ilpo
from data import SeqVisDemoDataset
from visual.data_aug import DataAugmentationBC
from visual.encoder_utils import build_visual_encoder

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

    print(f"Data loaded: there are {len(dataset)} demo frames.")

    # ============ building action quantizer ... ============
    # TODO: action_scale to be fine tuned for the task
    # Load model
    action_quantizer = ActionVQVAE(
        action_dim=3, 
        action_chunk_size=args.action_chunk_len,
        encoder_units=[16,16,16],
        decoder_units=[16,16,16],
        embedding_dim=16,
        codebook_size=32,
        decay=0.9,
        use_vq_layer=True,
    )

   # Load pretrained weights
    action_quantizer.load_state_dict(torch.load( \
        "/ssd01/gagan/cpt_checkpoints/jul14_vqvae_tabletop_v0.3/checkpoint.pth" \
        )["action_quantizer"])
    
    # Freeze weights and move to GPU
    action_quantizer.freeze()
    action_quantizer.cuda()

    # ============ building visual encoder network ... ============
    encoder, embed_dim = build_visual_encoder(args)

    encoder = utils.MultiCropWrapper(encoder)

    encoder = encoder.cuda()

    # ============ building policy network ... ============

    action_decoder = build_bet(args, input_dim=embed_dim)

    action_decoder = action_decoder.cuda()

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
    lr_schedule = utils.cosine_scheduler(
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
            "action_quantizer": action_quantizer.state_dict(),
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
    assert(action_quantizer.codebook_size == args.num_actions)
    cb_usage = torch.tensor([ 
        1308.,  1585.,   868.,  1143.,  1127.,  1981.,  2107.,   393., 14031.,
         816.,  1576.,  2229.,  2716.,  2265.,  1158.,  1469.,  1353.,    36.,
        1506.,  1873.,  1734.,    55.,  1208.,  1565.,  1129.,  1718.,    22.,
        1395.,  1394.,   828.,   797.,   631.])
    loss_weights = torch.div(torch.ones_like(cb_usage), cb_usage).unsqueeze(1).cuda()
    loss_weights = loss_weights / torch.sum(loss_weights) # (num_actions, 1)

    metric_logger = utils.MetricLogger(delimiter="  ")
    accuracies = torch.zeros(0).cuda()
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
        curr_embd = torch.stack(curr_embd, dim=1) # (batch_size, img_seq_len, embd_dim)
        goal_images = [im.cuda(non_blocking=True) for im in goal_images]
        goal_embd = torch.vstack(encoder(goal_images).chunk(args.naug + 1)).unsqueeze(1) # (batch_size, 1, embd_dim)

        actions = actions.repeat((args.naug + 1, 1, 1))
        amask = actions.repeat((args.naug + 1, 1, 1))

        # create one hot action vectors
        batch_size = curr_embd.shape[0]
        num_actions = action_decoder.num_actions
        _, idx, _ = action_quantizer(actions.cuda())
        onehot_actions = torch.zeros((batch_size, num_actions)).cuda()
        onehot_actions[torch.arange(batch_size), idx] = 1

        # predict logits_actions
        action_decoder_input = torch.cat([curr_embd, goal_embd], dim=1)
        logits_actions = action_decoder(action_decoder_input)

        # loss
        criterion = nn.CrossEntropyLoss(weight=loss_weights.squeeze(), reduction='mean')
        loss = criterion(logits_actions, onehot_actions)
        # loss = sigmoid_focal_loss(logits_actions, onehot_actions, reduction="none")  # (batch_size, num_actions)
        # loss = torch.mm(loss, loss_weights).mean()
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

        # compute batch accuracy, append to epoch accuracies
        pred_indices = torch.argmax(logits_actions, dim=1) # Change to use act() later
        true_indices = idx.cuda()
        accuracy = (torch.sum(pred_indices == true_indices)/batch_size).unsqueeze(dim=0)
        accuracies = torch.cat((accuracies, accuracy))

        # logging metrics
        torch.cuda.synchronize()
        metric_logger.update(action_loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
        metric_logger.update(accuracy=accuracies.mean())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CPT", parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_bc(args)
