import argparse

import numpy as np
import torch
from nav2d import Map2D, Robot
from nonvisual.models import ILPO


def test(args):

    print("Creating model...")
    model = ILPO(
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
    robot.reset()
    action_buffer = np.zeros((args.K, 2))
    for i in range(1000):
        goal = robot.goal.copy()
        obs = robot.pos.copy()
        model_input = [
            torch.tensor(tensor, dtype=torch.float32, device="cpu").unsqueeze(0)
            for tensor in [obs, goal]
        ]
        next_obs_pred, action, latent_actions = model(*model_input)
        action_buffer = np.vstack([action_buffer[1:, :], np.zeros((1, 2))])
        action = action.squeeze(0)
        action = action.detach().cpu().numpy()
        action_buffer = 0.1 * action_buffer + 0.9 * action

        print("action:", action_buffer[0])
        obs = robot.step(action_buffer[0])
        print("pred_obs:", next_obs_pred[0].detach().cpu().numpy())
        print("true_obs:", obs)

        if np.linalg.norm(obs[:2] - goal) < 0.1:
            print("Goal reached!")
            # No need to call reset() because robot.step() will call it


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="ilpo.pt")
    # TODO: Infer model hyperparameters from the model
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_hidden", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--K", type=int, default=10, help="Action chunk size")
    args = parser.parse_args()
    test(args)
