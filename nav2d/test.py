from nav2d import Map2D
from models import OIL
import numpy as np
import torch
import time
import matplotlib.pyplot as plt
import argparse
import time


class Robot:
    def __init__(self, map: Map2D) -> None:
        self.map = map
        self.pos = np.array([0, 0])
        self.goal = np.array([0, 0])

        # Rendering
        self.fig = None
        self.t = 0

    def step(self, action):
        assert action.shape == self.pos.shape, "action shape does not match"
        new_pos = self.pos + action
        if self.map._collision_check(new_pos):
            self.pos = new_pos

        self.t += 1
        self.render()
        if np.linalg.norm(self.pos - self.goal) < 0.1 or self.t > 50:
            self.reset()
        obs = np.concatenate([self.pos, self.goal])

        return obs

    def render(self):
        if self.fig is None:
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)

            # Plot arena
            arena = plt.Circle(self.map.arena_center, self.map.arena_radius, color="green")
            self.ax.add_artist(arena)

            # Plot obstacles
            for angle in self.map.obstacle_angles:
                obstacle_center = self.map.obstacle_center_radius * np.array(
                    [np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]
                )
                obstacle = plt.Circle(obstacle_center, self.map.obstacle_radius, color="black")
                self.ax.add_artist(obstacle)

            self.vis_robot = self.ax.plot(self.pos[0], self.pos[1], "o", color="red")[0]
            self.vis_goal = self.ax.plot(self.goal[0], self.goal[1], "o", color="blue")[0]

            self.ax.set_xlim([-self.map.arena_radius, self.map.arena_radius])
            self.ax.set_ylim([-self.map.arena_radius, self.map.arena_radius])
            self.ax.set_aspect("equal", adjustable="box")
            plt.show(block=False)

        self.vis_robot.set_data(self.pos)
        self.vis_goal.set_data(self.goal)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(0.05)

    def reset(self):
        while True:
            # Sample start and goal positions, until they are both valid
            angles = np.random.uniform(-180, 180, size=2)
            radius = np.random.uniform(0, self.map.arena_radius, size=2)
            start = radius[0] * np.array(
                [np.cos(np.deg2rad(angles[0])), np.sin(np.deg2rad(angles[0]))]
            )
            goal = radius[1] * np.array(
                [np.cos(np.deg2rad(angles[1])), np.sin(np.deg2rad(angles[1]))]
            )
            if self.map._collision_check(start) and self.map._collision_check(goal):
                self.pos = start
                self.goal = goal
                break
        obs = np.concatenate([self.pos, self.goal])
        self.t = 0
        return obs


def test(args):
    print("Creating model...")

    model = OIL(
        obs_dim=4,
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

    map = Map2D()
    # map.obstacle_radius = 0.125  # shrink obstacles to simulate padding
    robot = Robot(map)

    obs = robot.reset()
    action_buffer = np.zeros((10, 2))

    for i in range(1000):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device="cpu")
        obs_tensor = obs_tensor.unsqueeze(0)
        next_obs_pred, action = model(obs_tensor)
        action_buffer = np.vstack([action_buffer[1:, :], np.zeros((1, 2))])
        action = action.squeeze(0)
        action = action.detach().cpu().numpy()
        action_buffer = 0.1 * action_buffer + 0.9 * action

        print("action:", action_buffer[0])
        obs = robot.step(action_buffer[0])
        print("obs:", obs)

        if np.linalg.norm(obs[:2] - obs[2:]) < 0.1:
            print("Goal reached!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--num_hidden", type=int, default=2)
    parser.add_argument("--latent_dim", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--K", type=int, default=10, help="Action chunk size")
    args = parser.parse_args()
    test(args)
