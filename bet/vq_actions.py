import argparse, datetime, time, json, os
from pathlib import Path

import wandb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import torch
from torch import nn, Tensor
from torchvision import transforms
import torch.backends.cudnn as cudnn

from bet.model import MLP
from vector_quantize_pytorch import VectorQuantize
import visual.utils as utils
from visual.data_aug import DataAugmentationBC
from data import SeqVisDemoDataset
from bet.main_bet import get_args_parser


class ActionQuantizer(nn.Module):
    '''
    This class quantizes the continuous actions into discrete actions.

    action_dim:        Dimension of the action (3)
    action_chunk_size: Number of actions in an action chunk
    encoder_units:     The hidden layer nodes of encoder
    decoder_units:     The hidden layer nodes of decoder
    embedding_dim:     Dimension of the encoded action chunk 
    num_embeddings:    Number of quantized encoded action chunk

    '''
    def __init__(
        self,
        action_dim, 
        action_chunk_size, 
        encoder_units,
        decoder_units,
        embedding_dim,
        num_embeddings
    ) -> None:

        super().__init__()
        flat_input_dim = action_dim * action_chunk_size
        
        self.encoder   = MLP(flat_input_dim, embedding_dim, encoder_units)
        self.quantizer = VectorQuantize(embedding_dim, num_embeddings)
        self.decoder   = MLP(embedding_dim, flat_input_dim, decoder_units)

    def forward(self, x: Tensor):              # (batch_size, action_chunk_size, action_dim)
        x_flat =  x.flatten(start_dim=1)       # (batch_size, action_chunk_size*action_dim)

        z_e            = self.encoder(x_flat)  # (batch_size, embedding_dim)
        z_q, loss, idx = self.quantizer(z_e)   # (batch_size, embedding_dim)
        x_recon        = self.decoder(z_q)     # (batch_size, action_chunk_size*action_dim)

        x_recon_unflat = x_recon.view(x.shape) # (batch_size, action_chunk_size, action_dim)
        return x_recon_unflat


def train_vq(args):

    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    utils.wandb_init(args)

    # ============ getting data loader ... ============

    dataset = SeqVisDemoDataset(
        data_root=args.data_path,
        transform=transforms.ToTensor(),
        skip_frames=args.skip_frames,
        action_only=True,
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

    # ============ building action quantizer ... ============

    action_quantizer = ActionQuantizer(
        action_dim=3, 
        action_chunk_size=args.action_chunk_len,
        encoder_units=[16,16,32,64], 
        decoder_units=[8,8,16,16], 
        embedding_dim=64,
        num_embeddings=32,
    )
    action_quantizer = action_quantizer.cuda()

    # ============ preparing optimizer ... ============

    params_groups = utils.get_params_groups(nn.ModuleList([action_quantizer]))
    if args.optimizer == "adamw":
        optimizer = torch.optim.AdamW(params_groups)  # to use with ViTs
    elif args.optimizer == "sgd":
        optimizer = torch.optim.SGD(params_groups, lr=0, momentum=0.9)  # lr is set by scheduler
    elif args.optimizer == "lars":
        optimizer = utils.LARS(params_groups)  # to use with convnet and large batches
    
    # ============ init schedulers ... ============

    lr_schedule = utils.constant_scheduler(
        args.lr,
        args.epochs,
        len(data_loader)
    )
    wd_schedule = utils.constant_scheduler(
        args.weight_decay,
        args.epochs,
        len(data_loader)
    )

    # ============ start training ... ============

    print("Starting VQ training !")
    start_time = time.time()

    for epoch in range(0, args.epochs):
        data_loader.sampler.set_epoch(epoch)
        # ============ training one epoch of BC ... ============
        train_stats, action_pairs = train_one_epoch(
            data_loader,
            action_quantizer,
            optimizer,
            lr_schedule,
            wd_schedule,
            epoch,
            args,
        )

        # ============ writing logs ... ============

        # Save checkpoint.pth
        save_dict = {
            "action_quantizer": action_quantizer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
        }
        # Save checkpoint{epoch:04}.pth
        utils.save_on_master(save_dict, os.path.join(args.output_dir, "checkpoint.pth"))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f"checkpoint{epoch:04}.pth"))
        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}
        # Save log
        if utils.is_main_process():
            # log.txt
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            # wandb log
            utils.wandb_log(train_stats, epoch=epoch)
            # wandb image
            file_path  = os.path.join(args.output_dir, 'plots', f'{epoch:04}.png')
            save_2d_plot_one_epoch(action_pairs, file_path, num_pairs=10)
            wandb.log({'action_plot': wandb.Image(file_path)}, step=epoch)
            

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print("Training time {}".format(total_time_str))
    wandb.finish()


def save_3d_plot_one_epoch(action_pairs, file_path, num_pairs=1) -> Axes:
    '''
    plot actions and actions_recon onto a plot

    action_pairs: [[actions, actions_recon], [actions, actions_recon], ...] of shape (dataset_len, 2, action_chunk_size, 3)
    num_pairs:    Number of [actions, actions_recon] to plot; each pair is two curves

    actions:       Tensor of shape (action_chunk_size, 3)
    actions_recon: Tensor of shape (action_chunk_size, 3)
    '''
    # Create figure
    fig = plt.figure(dpi=100)
    ax: Axes = fig.add_subplot(projection='3d')

    # Plot
    for p in range(num_pairs):
        # Get a pair
        actions, actions_recon = action_pairs[action_pairs.shape[0]-1-p]
        assert(actions.shape == actions_recon.shape)
        # Get points to plot
        actions_cumu       = torch.cumsum(actions,       dim=0).cpu().detach().numpy().T # (3, action_chunk_size)
        actions_recon_cumu = torch.cumsum(actions_recon, dim=0).cpu().detach().numpy().T # (3, action_chunk_size)
        # Plot curves
        crv0, = ax.plot(actions_cumu      [0], actions_cumu      [1], actions_cumu      [2], \
                zdir='z', label=f'actions {p:02}')
        crv0, = ax.plot(actions_recon_cumu[0], actions_recon_cumu[1], actions_recon_cumu[2], \
                zdir='z', label=f'actions_recon {p:02}', color=crv0.get_color())
    
    # Beautify figure
    x_min, x_max = -0.055, 0.055
    y_min, y_max = -0.070, 0.070
    z_min, z_max = -0.055, 0.055

    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_zlim([z_min, z_max])
    ax.grid(False)

    # Save figure
    Path(file_path).parent.mkdir(exist_ok=True)
    plt.savefig(file_path)
    plt.close(fig)
    return fig


def save_2d_plot_one_epoch(action_pairs, file_path, num_pairs=1) -> Axes:
    '''
    plot actions and actions_recon onto a plot

    action_pairs: [[actions, actions_recon], [actions, actions_recon], ...] of shape (dataset_len, 2, action_chunk_size, 3)
    num_pairs:    Number of [actions, actions_recon] to plot; each pair is two curves

    actions:       Tensor of shape (action_chunk_size, 3)
    actions_recon: Tensor of shape (action_chunk_size, 3)
    '''
    # Create figure
    fig = plt.figure(figsize=(19.2, 4.8), dpi=100)
    ax_xy: Axes = fig.add_subplot(1, 3, 1)
    ax_yz: Axes = fig.add_subplot(1, 3, 2)
    ax_zx: Axes = fig.add_subplot(1, 3, 3)
    colormap = plt.get_cmap('rainbow')
    palette = [colormap(1.*p/num_pairs) for p in range(num_pairs)]

    # Plot
    for p in range(num_pairs):
        # Get a pair
        actions, actions_recon = action_pairs[action_pairs.shape[0]-1-p]
        assert(actions.shape == actions_recon.shape)
        # Get points to plot
        actions_cumu       = torch.cumsum(actions,       dim=0).cpu().detach().numpy().T # (3, action_chunk_size)
        actions_recon_cumu = torch.cumsum(actions_recon, dim=0).cpu().detach().numpy().T # (3, action_chunk_size)
        # Plot curves
        ax_xy.plot(actions_cumu      [0] , actions_cumu      [1]    , \
            label=f'actions {p:02}'      , color=palette[p])
        ax_xy.plot(actions_recon_cumu[0] , actions_recon_cumu[1]    , \
            label=f'actions_recon {p:02}', color=palette[p])
        ax_xy.plot([actions_cumu  [0,-1] , actions_recon_cumu[0,-1]], \
                   [actions_cumu  [1,-1] , actions_recon_cumu[1,-1]], \
                                      ':', color=palette[p])

        ax_yz.plot(actions_cumu      [1] , actions_cumu      [2]    , \
            label=f'actions {p:02}'      , color=palette[p])
        ax_yz.plot(actions_recon_cumu[1] , actions_recon_cumu[2]    , \
            label=f'actions_recon {p:02}', color=palette[p])
        ax_yz.plot([actions_cumu  [1,-1] , actions_recon_cumu[1,-1]], \
                   [actions_cumu  [2,-1] , actions_recon_cumu[2,-1]], \
                                      ':', color=palette[p])
            
        ax_zx.plot(actions_cumu      [2] , actions_cumu      [0]    , \
            label=f'actions {p:02}'      , color=palette[p])
        ax_zx.plot(actions_recon_cumu[2] , actions_recon_cumu[0]    , \
            label=f'actions_recon {p:02}', color=palette[p])
        ax_zx.plot([actions_cumu  [2,-1] , actions_recon_cumu[2,-1]], \
                   [actions_cumu  [0,-1] , actions_recon_cumu[0,-1]], \
                                      ':', color=palette[p])
    
    # Beautify figure
    x_min, x_max = -0.055, 0.055
    y_min, y_max = -0.070, 0.070
    z_min, z_max = -0.055, 0.055

    ax_xy.set_xlim([x_min, x_max])
    ax_xy.set_ylim([y_min, y_max])
    ax_xy.set_xlabel('X')
    ax_xy.set_ylabel('Y')
    ax_xy.grid(False)
    ax_yz.set_xlim([y_min, y_max])
    ax_yz.set_ylim([z_min, z_max])
    ax_yz.set_xlabel('Y')
    ax_yz.set_ylabel('Z')
    ax_yz.grid(False)
    ax_zx.set_xlim([z_min, z_max])
    ax_zx.set_ylim([x_min, x_max])
    ax_zx.set_xlabel('Z')
    ax_zx.set_ylabel('X')
    ax_zx.grid(False)

    # Save figure
    Path(file_path).parent.mkdir(exist_ok=True)
    plt.savefig(file_path)
    plt.close(fig)
    return fig


def train_one_epoch(
    data_loader,
    action_quantizer,
    optimizer,
    lr_schedule,
    wd_schedule,
    epoch,
    args,
):
    metric_logger = utils.MetricLogger(delimiter="  ")
    action_logger = torch.empty(0, 2, 6, 3).cuda()
    header = "Epoch: [{}/{}]".format(epoch, args.epochs)
    for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):

        actions, amask = batch
        actions = actions.cuda()

        # update weight decay and learning rate according to their schedule
        it = len(data_loader) * epoch + it  # global training iteration
        for i, param_group in enumerate(optimizer.param_groups):
            param_group["lr"] = lr_schedule[it]
            if i == 0:  # only the first group is regularized
                param_group["weight_decay"] = wd_schedule[it]

        # forward pass: encode and decode to get reconstructed actions
        actions_recon = action_quantizer(actions)
        
        # loss
        criterion = get_loss
        loss, actions_cumu, actions_recon_cumu = criterion(actions_recon, actions)

        # optimizer step
        optimizer.zero_grad()
        loss.backward()
        # param_norms = None
        # if args.clip_grad:
        #     param_norms = utils.clip_gradients(encoder, args.clip_grad)
        # utils.cancel_gradients_last_layer(epoch, encoder, args.freeze_last_layer)
        optimizer.step()

        # logging metrics
        metric_logger.update(action_loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])

        # logging actions
        action_logger = torch.cat((action_logger,torch.stack((actions,actions_recon),dim=1)))

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}, action_logger


def get_loss(actions_recon: Tensor, actions: Tensor):
    '''
    actions_recon: Reconstructed actions of shape (batch_size, action_chunk_size, 3)
    actions      : Ground truth  actions of shape (batch_size, action_chunk_size, 3)
    '''
    actions_cumu       = torch.cumsum(actions,       dim=1) # (batch_size, action_chunk_size, 3)
    actions_recon_cumu = torch.cumsum(actions_recon, dim=1) # (batch_size, action_chunk_size, 3)

    criterion = nn.MSELoss()
    loss = criterion(actions_recon_cumu, actions_cumu)

    return loss, actions_cumu, actions_recon_cumu


if __name__ == "__main__":
    parser = argparse.ArgumentParser("CPT", parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_vq(args)