from dataloader import Nav2DDataloader

dataloader = Nav2DDataloader(seq_len=2, batch_size=32, shuffle=True, num_workers=2)

import torch
from models import OIL
from torch import optim
from torch import nn

import wandb

def wandb_init():
    wandb.init(project="CPT", tags="nav2d", name="OIL")


def train():
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
            batch  = [b.to(dtype=torch.float32, device='cuda') for b in batch]
            batch_obs, batch_actions = batch
            batch_actions = batch_actions[:, 0, :] # TODO: Fix this hack
            curr_obs = batch_obs[:, 0, :]
            next_obs = batch_obs[:, 1, :]

            next_obs_pred, actions = model(curr_obs)

            # OIL loss
            loss_oil = mse(next_obs_pred, next_obs) + mse(actions, batch_actions)
            optimizer_oil.zero_grad()
            loss_oil.backward()
            optimizer_oil.step()

            # Action prediction loss
            # print("Action shape:", actions.shape)
            # print("Batch actions shape:", batch_actions.shape)

            loss_action = mse(actions, batch_actions)
            # optimizer_action.zero_grad()
            # loss_action.backward()
            # optimizer_action.step()

            # loss_action = torch.tensor(0.0)
            
            # Log loss
            print(f'Epoch: {epoch}, OIL loss: {loss_oil.item()}, Action loss: {loss_action.item()}')
            wandb.log({'OIL loss': loss_oil.item(), 'Action loss': loss_action.item()})
    
    
    torch.save(model.state_dict(), 'oil.pt')

    # TODO: Measure mutual information between 
            

def main():
    wandb_init()
    train()
    wandb.finish()

    
if __name__ == '__main__':
    main()
