import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import umap
import umap.plot
from vector_quantize_pytorch.cartesian_quantize import CartesianActionChunkQuantize
from visual import utils

import wandb


def log_recons(encoder, decoder, data_loader, epoch, args):

    nsample = 8

    for batch in data_loader:
        o_curr, o_next, o_goal, actions, amask = batch

        # keep only 8 samples
        o_curr = o_curr[:nsample].cuda()
        o_next = o_next[:nsample].cuda()
        o_goal = o_goal[:nsample].cuda()

        x_next_pred, _, z_curr, z_reg_loss, x_reg_loss = encoder(o_curr, o_next, o_goal)
        o_next_pred = decoder(x_next_pred)
        recons_out = torch.concat([o_curr, o_next, o_next_pred], dim=-1)
        for i in range(nsample):
            utils.save_img(recons_out[i], os.path.join(args.output_dir, f"ep{epoch}im{i}"))

        break


def umap_and_log(data, labels, epoch, args, suffix="umap"):
    reducer = umap.UMAP()

    u = reducer.fit_transform(data)
    umap.plot.points(reducer, labels=labels, theme="fire")
    fn = os.path.join(args.output_dir, f"ep{epoch}_{suffix}.png")
    plt.savefig(fn)
    wandb.log({suffix: wandb.Image(fn)}, step=epoch)


def log_latent_umap(encoder, data_loader, epoch, args):
    """Collect data over the entire dataset and log the UMAP embedding of the latent space."""

    # catesian action quantizer that is specific to table top reorientation task
    # action_quantizer = CartesianActionChunkQuantize(num_actions=7, action_scale=0.0008)

    nsample = 512

    x_currs = []
    z_currs = []
    embodiment_labels = []
    action_labels = []
    k = 0
    for it, batch in enumerate(data_loader):
        o_curr, o_next, o_goal, actions, amask = batch

        if args.core == "ilpo":
            o_curr = o_curr.cuda()
            o_goal = o_goal.cuda()
            o_next = None
        else:
            o_curr = o_curr.cuda()
            o_next = o_next.cuda()
            o_goal = None

        x_next_pred, x_curr, z_curr, z_reg_loss, x_reg_loss = encoder(o_curr, o_next, o_goal)

        # TODO: Return x_curr from the encoder
        x_currs.append(x_curr.detach().cpu().numpy())
        z_currs.append(z_curr.detach().cpu().numpy())
        # Infer embodiement from amask
        # amask is a zero-tensor for human demonstrations and a one-tensor for robot demonstrations
        embodiment_labels.append(amask[:, 0, 0].detach().cpu().numpy())  # appends [0, 1, 0, ... batch_size] vector
        # action_labels.append(action_quantizer(actions)[1].detach().cpu().numpy())

        k += o_curr.shape[0]
        if k > nsample:
            break

    x_currs = np.concatenate(x_currs, axis=0)
    z_currs = np.concatenate(z_currs, axis=0)
    embodiment_labels = np.concatenate(embodiment_labels, axis=0)
    # action_labels = np.concatenate(action_labels, axis=0)

    umap_and_log(x_currs, embodiment_labels, epoch, args, suffix="umap_x")
    umap_and_log(z_currs, embodiment_labels, epoch, args, suffix="umap_z")
