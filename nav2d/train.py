from dataloader import Nav2DDataloader


import torch
from models import OIL
from torch import optim
from torch import nn

import wandb
import friendlywords as fw
from datetime import datetime


def wandb_init():
    date_time = datetime.now()
    name = "OIL" + "-" + fw.generate(1) + "-" + date_time.strftime("%m-%d")
    wandb.init(project="CPT", tags="nav2d", name=name)


def train():
    # Create dataloader
    dataloader = Nav2DDataloader(seq_len=2, batch_size=32, shuffle=True, num_workers=2)

    # Create model and optimizer
    model = OIL(obs_dim=4, act_dim=2, hidden_dim=64, num_hidden=2)
    model = model.cuda()
    optimizer_oil = optim.Adam(model.parameters(), lr=1e-3)
    optimizer_action = optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    # Train model
    loss = torch.inf
    for epoch in range(10):
        for i, batch in enumerate(dataloader):
            batch = [b.to(dtype=torch.float32, device="cuda") for b in batch]
            batch_obs, batch_actions = batch
            batch_actions = batch_actions[:, 0, :]  # TODO: Fix this hack
            curr_obs = batch_obs[:, 0, :]
            next_obs = batch_obs[:, 1, :]

            next_obs_pred, actions = model(curr_obs)

            # OIL loss
            obs_pred_loss = mse(next_obs_pred, next_obs)
            action_pred_loss = mse(actions, batch_actions)
            loss_oil = obs_pred_loss + action_pred_loss
            optimizer_oil.zero_grad()
            loss_oil.backward()
            optimizer_oil.step()

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


def main():
    wandb_init()
    train()
    wandb.finish()


if __name__ == "__main__":
    main()
