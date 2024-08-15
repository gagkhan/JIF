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

import visual.utils as utils
from bet.utils import build_bet, build_action_decoder
from bet.vq_actions import ActionVQVAE
from bet.args_parser import get_args_parser
from cpt import ilpo
from data import load_dataset
from visual.data_aug import DataAugmentationBC
from visual.encoder_utils import build_visual_encoder


def train_bet(args):

    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    utils.wandb_init(args)

    # ============ Get dataloaders ... ============
    transform = DataAugmentationBC(args.naug)

    dataset, val_dataset = load_dataset(args, wrapper_cls="SeqVisDemoDataset", transform=transform)
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

    print(f"Data loaded: there are {len(dataset)} demo frames.")

    # ============ building action quantizer ... ============
    state_dict = torch.load("/ssd01/gagan/cpt_checkpoints/jul14_vqvae_tabletop_v0.3/checkpoint.pth" )
    training_args = state_dict["args"]
    
    # Load model
    action_quantizer = ActionVQVAE(
        action_dim=3, 
        action_chunk_len
                     =training_args.action_chunk_len,
        encoder_units=training_args.action_quantizer_encoder_units,
        decoder_units=training_args.action_quantizer_decoder_units,
        embedding_dim=training_args.action_quantizer_embedding_dim,
        codebook_size=training_args.num_actions,
        decay        =training_args.action_quantizer_decay,
        use_vq_layer =training_args.action_quantizer_use_vq_layer,
    )

    # Store action quantizer training args into args
    assert(args.action_chunk_len == training_args.action_chunk_len)
    assert(args.num_actions      == training_args.num_actions)
    assert(True                  == training_args.action_quantizer_use_vq_layer)
    args.action_quantizer_use_vq_layer  = training_args.action_quantizer_use_vq_layer
    args.action_quantizer_encoder_units = training_args.action_quantizer_encoder_units
    args.action_quantizer_decoder_units = training_args.action_quantizer_decoder_units
    args.action_quantizer_embedding_dim = training_args.action_quantizer_embedding_dim
    args.action_quantizer_decay         = training_args.action_quantizer_decay

    # Load pretrained weights
    action_quantizer.load_state_dict(state_dict["action_quantizer"])

    # Load code weights. This will be used to weight the loss in bet training below
    action_quantizer.code_weights = state_dict["code_weights"].cuda()
    
    # Freeze weights and move to GPU
    action_quantizer.freeze()
    action_quantizer.cuda()

    # ============ building visual encoder network ... ============
    encoder, embed_dim = build_visual_encoder(args)

    encoder = utils.MultiCropWrapper(encoder)

    encoder = encoder.cuda()

    # ============ building latent policy network ... ============

    student = build_bet(args, input_img_dim=embed_dim)

    student = student.cuda()

    # ============ building action decoder network ... ============

    action_decoder = build_action_decoder(args, student.n_embd)
    
    action_decoder = action_decoder.cuda()

    # ============ preparing criterion ... ============
    criterion = torch.hub.load(
        'adeelh/pytorch-multi-class-focal-loss',
        model='FocalLoss',
        alpha=action_quantizer.code_weights,
        gamma=2,
        reduction='mean',
        force_reload=False,
        verbose=False)

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
        encoder=encoder,
        student=student,
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
            student,
            action_decoder,
            data_loader,
            action_quantizer,
            criterion,
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
                encoder,
                student,
                action_decoder,
                val_data_loader,
                action_quantizer,
                criterion,
                args,
            )

        epoch_stats = {**train_stats, **val_stats}

        # ============ writing logs ... ============
        save_dict = {
            "encoder": encoder.state_dict(),
            "student": student.state_dict(),
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
    student,
    action_decoder,
    data_loader,
    action_quantizer,
    criterion,
    optimizer,
    lr_schedule,
    wd_schedule,
    epoch,
    fp16_scaler,
    args,
):

    # prepare for training: put to train mode
    for m in [encoder, student, action_decoder]:
        m.train()
    
    metric_logger = utils.MetricLogger(delimiter="  ")
    accuracies = torch.zeros(0).cuda()
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)
    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        if args.use_ee:
            img_sequences, goal_images, ee_sequences, actions, amask = batch
        else:
            img_sequences, goal_images, actions, amask = batch
            ee_sequences = None

        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # move images to gpu, use only one global view for the goal
        #   curr_embd: ((naug+1)*batch_size, seq_len, embd_dim)
        #   goal_embd: ((naug+1)*batch_size,       1, embd_dim)
        curr_embd = []
        for img in img_sequences:
            img = [im.cuda(non_blocking=True) for im in img]
            curr_embd.append(torch.vstack(encoder(img).chunk(args.naug + 1)))
        curr_embd = torch.stack(curr_embd, dim=1)
        
        goal_images = [im.cuda(non_blocking=True) for im in goal_images]
        goal_embd = torch.vstack(encoder(goal_images).chunk(args.naug + 1))
        goal_embd = goal_embd.unsqueeze(1)

        # move ee_sequences to gpu
        if args.use_ee:
            ee_sequences = torch.stack(ee_sequences, dim=1).repeat((args.naug+1, 1, 1)).cuda()

        # create onehot_actions tensor
        actions = actions.repeat((args.naug+1, 1, 1))
        amask = amask.repeat((args.naug+1, 1, 1))

        batch_size = curr_embd.shape[0]
        num_actions = args.num_actions
        _, idx, _ = action_quantizer(actions.cuda())
        onehot_actions = torch.zeros((batch_size, num_actions)).cuda()
        onehot_actions[torch.arange(batch_size), idx] = 1

        # predict logits_actions
        logits_actions = action_decoder(student(curr_embd, goal_embd), ee_sequences)

        # loss
        loss = criterion(logits_actions, idx)
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
        accuracy = (torch.sum(pred_indices == true_indices)/batch_size).unsqueeze(0)
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


def validate(
    encoder,
    student,
    action_decoder,
    data_loader,
    action_quantizer,
    criterion,
    args,
):

    # prepare for validation: put to eval mode
    for m in [encoder, student, action_decoder]:
        m.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    accuracies = torch.zeros(0).cuda()
    header = "Validation: "
    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
        with torch.no_grad():
            if args.use_ee:
                img_sequences, goal_images, ee_sequences, actions, amask = batch
            else:
                img_sequences, goal_images, actions, amask = batch

            # move images to gpu, use only one global view for the goal
            #   curr_embd: ((naug+1)*batch_size, seq_len, embd_dim)
            #   goal_embd: ((naug+1)*batch_size,       1, embd_dim)
            curr_embd = []
            for img in img_sequences:
                img = [im.cuda(non_blocking=True) for im in img]
                curr_embd.append(torch.vstack(encoder(img).chunk(args.naug + 1)))
            curr_embd = torch.stack(curr_embd, dim=1)
            
            goal_images = [im.cuda(non_blocking=True) for im in goal_images]
            goal_embd = torch.vstack(encoder(goal_images).chunk(args.naug + 1))
            goal_embd = goal_embd.unsqueeze(1)

            # move ee_sequences to gpu
            if args.use_ee:
                ee_sequences = torch.stack(ee_sequences, dim=1).repeat((args.naug+1, 1, 1)).cuda()

            # create onehot_actions tensor
            actions = actions.repeat((args.naug+1, 1, 1))
            amask = amask.repeat((args.naug+1, 1, 1))

            batch_size = curr_embd.shape[0]
            num_actions = args.num_actions
            _, idx, _ = action_quantizer(actions.cuda())
            onehot_actions = torch.zeros((batch_size, num_actions)).cuda()
            onehot_actions[torch.arange(batch_size), idx] = 1

            # predict logits_actions
            logits_actions = action_decoder(student(curr_embd, goal_embd), ee_sequences)

            # loss
            loss = criterion(logits_actions, idx)
            if not math.isfinite(loss.item()):
                print("Loss is {}, stopping training".format(loss.item()), force=True)
                sys.exit(1)
        
            # compute batch accuracy, append to epoch accuracies
            pred_indices = torch.argmax(logits_actions, dim=1) # Change to use act() later
            true_indices = idx.cuda()
            accuracy = (torch.sum(pred_indices == true_indices)/batch_size).unsqueeze(dim=0)
            accuracies = torch.cat((accuracies, accuracy))

            # logging metrics
            torch.cuda.synchronize()
            metric_logger.update(val_action_loss=loss.item())
            metric_logger.update(val_accuracy=accuracies.mean())

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CPT", parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_bet(args)
