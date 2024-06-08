import argparse
import json
import os
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from data_utils import VisDemoDataset
from torch import optim
from visual import utils
from vqbet import VectorQuantization


def get_arg_parser():

    parser = argparse.ArgumentParser("VQ-BeT quantization stage")

    parser.add_argument(
        "--data_path",
        default="/path/to/demos/",
        type=str,
        help="Please specify path to the demonstration training data.",
    )
    parser.add_argument(
        "--skip_frames",
        default=5,
        type=int,
        help="Number of frames to skip when loading the dataset.",
    )

    parser.add_argument("--embed_dim", default=4, type=int, help="Dimensionality of embedding")
    parser.add_argument("--codebook_len", default=32, type=int, help="Number of codes (vectors) in the codebook")

    parser.add_argument("--batch_size", default=128, type=int, help="Batch size for training")
    parser.add_argument("--lr", default=0.0001, type=float, help="Batch size for training")
    parser.add_argument("--epochs", default=100, type=int, help="Number of epochs of training.")
    parser.add_argument("--disable_wnb", default=False, type=utils.bool_flag, help="Disable wandb logging.")

    parser.add_argument("--output_dir", default=".", type=str, help="Path to save logs and checkpoints.")
    parser.add_argument("--saveckp_freq", default=1000, type=int, help="Save checkpoint every x epochs.")
    parser.add_argument("--seed", default=0, type=int, help="Random seed.")
    parser.add_argument("--num_workers", default=10, type=int, help="Number of data loading workers per GPU.")

    parser.add_argument(
        "--pretrained_weights",
        default="",
        type=str,
        help="Path to pretrained weights to load before training.",
    )

    return parser


def train_vq(args):

    # utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v)) for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    utils.wandb_init(args)

    dataset = VisDemoDataset(
        data_root=args.data_path,
        skip_frames=args.skip_frames,
        action_only=True,
    )
    # sampler = torch.utils.data.DistributedSampler(dataset, shuffle=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        # sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    action_shape = dataset.action_shape

    vq_model = VectorQuantization(
        input_dim=action_shape[0] * action_shape[1],
        embed_dim=args.embed_dim,
        codebook_len=args.codebook_len,
    )
    vq_model.to(device="cuda:0")

    lr_schedule = [args.lr * (1 - e / args.epochs) for e in range(args.epochs)]
    optimizer = optim.AdamW(vq_model.parameters(), lr=lr_schedule[0])

    for epoch in range(args.epochs):

        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Epoch: [{}/{}]".format(epoch, args.epochs)

        for it, batch in enumerate(metric_logger.log_every(data_loader, 50, header)):
            actions, amask = batch
            actions = actions.to(device="cuda:0")

            loss, recons_loss, vq_loss = vq_model(actions.reshape(-1, action_shape[0] * action_shape[1]))

            optimizer.zero_grad()
            loss = recons_loss + 0.1 * vq_loss
            loss.backward()
            optimizer.step()

            metric_logger.update(loss=loss.item())
            metric_logger.update(recons_loss=recons_loss.item())
            metric_logger.update(vq_loss=vq_loss.item())

        save_dict = {
            "vq_model": vq_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch + 1,
            "args": args,
        }

        utils.save_on_master(save_dict, os.path.join(args.output_dir, "checkpoint.pth"))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(args.output_dir, f"checkpoint{epoch:04}.pth"))

        train_stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}
        log_stats = {**{f"train_{k}": v for k, v in train_stats.items()}, "epoch": epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")
            utils.wandb_log(train_stats, epoch=epoch)

        for param_group in optimizer.param_groups:
            param_group["lr"] = lr_schedule[epoch]


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_vq(parser.parse_args())
