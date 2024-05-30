import argparse
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from data_utils import VisDemoDataset
from visual import utils
from vqbet import ResidualVQ


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

    parser.add_argument("--batch_size", default=32, help="Batch size for training")

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

    vq_model = ResidualVQ(input_dim=action_shape[0] * action_shape[1], embed_dim=4, codebook_len=16)

    for epoch in range(args.epochs):

        metric_logger = utils.MetricLogger(delimiter="  ")
        header = "Epoch: [{}/{}]".format(epoch, args.epochs)

        for it, batch in enumerate(metric_logger.log_every(data_loader, 10, header)):
            actions, amask = batch

            _, loss = vq_model(actions.reshape(-1, action_shape[0] * action_shape[1]))

            loss.backward()


if __name__ == "__main__":
    parser = get_arg_parser()
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train_vq(parser.parse_args())
