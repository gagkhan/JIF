from dataloader import Nav2DDataloader

dataloader = Nav2DDataloader(seq_len=2, batch_size=32, shuffle=True, num_workers=2)

import torch
from models import OIL
from torch import optim
from torch import nn


def main():
    # Create model and optimizer
    model = OIL(obs_dim=4, act_dim=2, hidden_dim=64, num_hidden=2)
    model = model.cuda()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    # Train model
    loss = torch.inf
    for epoch in range(2):
        for i, batch in enumerate(dataloader):
            batch  = [b.to(dtype=torch.float32, device='cuda') for b in batch]
            batch_obs, batch_actions = batch
            curr_obs = batch_obs[:, 0, :]
            next_obs = batch_obs[:, 1, :]
            next_obs_pred, actions = model(curr_obs)
            loss = mse(next_obs_pred, next_obs)
            if i == 0 and epoch == 0:
                print(f'Epoch: {epoch}, Loss: {loss.item()}')
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            print(f'Epoch: {epoch}, Loss: {loss.item()}')

    
if __name__ == '__main__':
    main()
