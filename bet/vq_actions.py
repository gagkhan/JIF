import argparse, datetime, time, json, os
from pathlib import Path

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
    embedding_dim:     Dimension of the encoded action chunk 
    num_embeddings:    Number of quantized encoded action chunk
    '''
    def __init__(
        self,
        action_dim, 
        action_chunk_size, 
        embedding_dim,
        num_embeddings
    ) -> None:

        super().__init__()
        flat_input_dim = action_dim * action_chunk_size
        
        self.encoder   = MLP(flat_input_dim, embedding_dim, [])
        self.quantizer = VectorQuantize(embedding_dim, num_embeddings)
        self.decoder   = MLP(embedding_dim, flat_input_dim, [])

    def forward(self, x: Tensor):              # (batch, action_chunk_size, action_dim)
        x_flat =  x.flatten(start_dim=1)       # (batch, action_chunk_size*action_dim)

        z_e          = self.encoder(x_flat)    # (batch, embedding_dim)
        z_q, loss, _ = self.quantizer(z_e)     # (batch, embedding_dim)
        x_recon      = self.decoder(z_q)       # (batch, action_chunk_size*action_dim)

        x_recon_unflat = x_recon.view(x.shape) # (batch, action_chunk_size, action_dim)
        return x_recon_unflat, loss


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
        embedding_dim=16,
        num_embeddings=8,
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
        train_stats = train_one_epoch(
            data_loader,
            action_quantizer,
            optimizer,
            lr_schedule,
            wd_schedule,
            epoch,
            args,
        )

        # ============ writing logs ... ============
        save_dict = {
            "action_quantizer": action_quantizer.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
        }
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
    data_loader,
    action_quantizer,
    optimizer,
    lr_schedule,
    wd_schedule,
    epoch,
    args,
):

    metric_logger = utils.MetricLogger(delimiter="  ")
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
        criterion = nn.MSELoss()
        loss = criterion(actions_recon, actions)

        # optimizer step
        optimizer.zero_grad()
        param_norms = None
        loss.backward()
        # if args.clip_grad:
        #     param_norms = utils.clip_gradients(encoder, args.clip_grad)
        # utils.cancel_gradients_last_layer(epoch, encoder, args.freeze_last_layer)
        optimizer.step()

        # logging
        metric_logger.update(action_loss=loss.item())
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
    train_vq(args)