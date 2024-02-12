import argparse
from datetime import datetime

import friendlywords as fw
import torch
from dataloader import Nav2DDataloader
from models import OIL
from torch import nn, optim

import wandb


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
    alpha = args.alpha
    beta = args.beta

    # Create dataloader
    dataloader = Nav2DDataloader(
        skip_frames=K - 1,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    # Create model and optimizer
    model = OIL(
        obs_dim=2,
        goal_dim=2,
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
            batch_obs, batch_obs_next, batch_goals, batch_actions = batch
            next_obs_pred, actions, latent_actions = model(batch_obs, batch_goals)

            # OIL loss
            obs_pred_loss = mse(next_obs_pred, batch_obs_next)
            action_pred_loss = mse(actions, batch_actions)
            latent_reg_loss = torch.linalg.norm(latent_actions, dim=-1).mean()
            loss_oil = obs_pred_loss + alpha * action_pred_loss + beta * latent_reg_loss
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


def main(args):
    wandb_init()
    train(args)
    wandb.finish()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--detach_latent", action="store_true")
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--K", type=int, default=10, help="Action chunk size")
    args = parser.parse_args()
    main(args)
