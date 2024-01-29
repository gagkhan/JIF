from dataloader import Nav2DDataloader


import torch
from models import OIL
from torch import optim
from torch import nn

import wandb
import friendlywords as fw
from datetime import datetime

import argparse


def wandb_init():
    date_time = datetime.now()
    name = "OIL" + "-" + fw.generate(1) + "-" + date_time.strftime("%m-%d")
    wandb.init(project="CPT", tags="nav2d", name=name)


def train(args):
    batch_size = args.batch_size
    num_workers = args.num_workers
    detach_latent = args.detach_latent
    latent_dim = args.latent_dim
    lr = args.lr
    max_epochs = args.epochs
    K = args.K
    seq_len = K + 1

    # Create dataloader
    dataloader = Nav2DDataloader(
        seq_len=seq_len,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # Create model and optimizer
    model = OIL(
        obs_dim=4,
        act_dim=2,
        hidden_dim=64,
        num_hidden=2,
        latent_dim=latent_dim,
        detach_latent=detach_latent,
        action_chunck=K,
    )
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    mse = nn.MSELoss()

    # Train model
    loss = torch.inf
    for epoch in range(max_epochs):
        for i, batch in enumerate(dataloader):
            batch = [b.to(dtype=torch.float32, device="cuda") for b in batch]
            batch_obs, batch_actions = batch
            batch_actions = batch_actions[:, :K, :]  # TODO: Fix this hack

            curr_obs = batch_obs[:, 0, :]

            # observation after K steps
            next_obs = batch_obs[:, K, :]

            next_obs_pred, actions = model(curr_obs)

            # OIL loss
            obs_pred_loss = mse(next_obs_pred, next_obs)
            action_pred_loss = mse(actions, batch_actions)
            loss_oil = obs_pred_loss + action_pred_loss
            optimizer.zero_grad()
            loss_oil.backward()
            optimizer.step()

            # Log loss
            print(
                f"Epoch: {epoch}, OIL loss: {loss_oil.item()}, Obs loss: {obs_pred_loss.item()}, Action loss: {action_pred_loss.item()}"
            )
            wandb.log(
                {
                    "OIL loss": loss_oil.item(),
                    "Obs loss": obs_pred_loss.item(),
                    "Action loss": action_pred_loss.item(),
                }
            )

    torch.save(model.state_dict(), "oil.pt")

    # TODO: Measure mutual information between


def main(args):
    wandb_init()
    train(args)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--detach_latent", action="store_true")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--K", type=int, default=1, help="Action chunk size")
    args = parser.parse_args()
    main(args)
