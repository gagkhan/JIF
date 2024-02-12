import argparse

import numpy as np
import torch
from models import OIL
from nav2d import Map2D, Robot


def test(args):
    print("Creating model...")

    model = OIL(
        obs_dim=2,
        goal_dim=2,
        act_dim=2,
        hidden_dim=args.hidden_dim,
        num_hidden=args.num_hidden,
        latent_dim=args.latent_dim,
        action_chunck=args.K,
    )

    print("Loading weights...")

    model.load_state_dict(torch.load("oil.pt", map_location="cpu"))
    model.eval()

    print("Model loaded.")

    # Create map and robot
    print("Creating map and robot...")
    map = Map2D()
    # map.obstacle_radius = 0.125  # shrink obstacles to simulate padding
    robot = Robot(map)

    
    print("Testing model...")
    obs, goal = robot.reset()
    action_buffer = np.zeros((10, 2))
    for i in range(1000):
        # obs_tensor = torch.tensor(obs, dtype=torch.float32, device="cpu")
        model_input = np.concatenate([obs, goal])
        obs_tensor = torch.tensor(model_input, dtype=torch.float32, device="cpu")
        obs_tensor = obs_tensor.unsqueeze(0)
        next_obs_pred, action = model(obs_tensor)
        action_buffer = np.vstack([action_buffer[1:, :], np.zeros((1, 2))])
        action = action.squeeze(0)
        action = action.detach().cpu().numpy()
        action_buffer = 0.1 * action_buffer + 0.9 * action

        print("action:", action_buffer[0])
        obs = robot.step(action_buffer[0])
        print("pos:", next_obs_pred[0].detach().cpu().numpy() - obs)
        print("obs:", obs)

        if np.linalg.norm(obs[:2] - goal) < 0.1:
            print("Goal reached!")
            # No need to call reset() because robot.step() will call it


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="oil.pt")
    # TODO: Infer model hyperparameters from the model
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_hidden", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--K", type=int, default=10, help="Action chunk size")
    args = parser.parse_args()
    test(args)
