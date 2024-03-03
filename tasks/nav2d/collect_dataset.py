import argparse
import os
import pickle
from datetime import datetime

import numpy as np
from nav2d import Map2D, Robot, RRTExpert
from tqdm import tqdm

np.random.seed(605)


def collect_demo_dataset(num_demo=1000, visual=False):

    # # current date and time
    # date_time = datetime.now()
    # date_time = date_time.strftime("%Y-%m-%d-%H-%M-%S")

    # Prepare directory to save visual data

    if visual:
        basename = "nav2d_visual"
    else:
        basename = "nav2d"
    outdir = os.path.join(os.environ["DATA_ROOT"], basename)
    os.system(f"rm -rf {outdir}")
    os.makedirs(outdir, exist_ok=True)

    map = Map2D()
    expert = RRTExpert(map)

    robot = Robot(map)
    robot.set_render_sleep(0.0)
    robot.reset()
    robot.auto_reset = False
    demos = []
    for i in tqdm(range(num_demo)):
        angles = np.random.uniform(-180, 180, size=2)
        radius = np.random.uniform(0, map.arena_radius, size=2)
        start = radius[0] * np.array([np.cos(np.deg2rad(angles[0])), np.sin(np.deg2rad(angles[0]))])
        goal = radius[1] * np.array([np.cos(np.deg2rad(angles[1])), np.sin(np.deg2rad(angles[1]))])
        if expert._is_valid_node(start) and expert._is_valid_node(goal):
            expert.plan(start, goal)
            path, actions = expert._get_path()

            assert np.linalg.norm(path[0] - start) < 0.1, f"start: {start}, path[0]: {path[0]}"
            assert np.linalg.norm(path[-1] - goal) < 0.1, f"goal: {goal}, path[-1]: {path[-1]}"

            if not visual:
                # Just append the data to demos, no need to render
                # All the demos are saved at once at the end
                demos.append((start, goal, path, actions))
            else:
                # visual observations are needed
                # To save visual data, we need to render the robot and save the images

                # First, lets configure start, goal and robot position
                robot.goal = goal
                robot.start = start
                robot.pos = start
                robot.render()
                os.makedirs(f"{outdir}/{i}", exist_ok=True)
                # Then, follow path from RRT expert and save images
                for j, action in enumerate(actions):
                    robot.render()
                    robot.fig.savefig(f"{outdir}/{i}/{j:06}.png")
                    robot.step(action)
                # save the last image
                robot.render()
                robot.fig.savefig(f"{outdir}/{i}/{j+1:06}.png")

                # save actions
                np.save(f"{outdir}/actions.npy", actions)
                # save path
                np.save(f"{outdir}/path.npy", path)

    if not visual:
        pickle.dump(demos, open(f"{outdir}/demos.pkl", "wb"))


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_demo", type=int, default=1000)
    parser.add_argument("--visual", default=False, action="store_true")
    args = parser.parse_args()

    collect_demo_dataset(args.num_demo, args.visual)
