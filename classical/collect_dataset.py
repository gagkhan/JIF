import os

import argparse
import numpy as np
from tqdm import tqdm

from nav2d import Map2D
from nav2d import RRTExpert

from datetime import datetime

import pickle



def collect_demo_dataset(num_demo=1000):
    map = Map2D()
    expert = RRTExpert(map)
    demos = []
    for i in tqdm(range(num_demo)):
        angles = np.random.uniform(-180, 180, size=2)
        radius = np.random.uniform(0, map.arena_radius, size=2)
        start = radius[0] * np.array([ np.cos(np.deg2rad(angles[0])), np.sin(np.deg2rad(angles[0]))])
        goal = radius[1] * np.array([ np.cos(np.deg2rad(angles[1])), np.sin(np.deg2rad(angles[1]))])
        if expert._is_valid_node(start) and expert._is_valid_node(goal):
            expert.plan(start, goal)
            path, path_edges = expert._get_path()
            demos.append((start, goal, path, path_edges))
    # current date and time
    date_time = datetime.now()
    date_time = date_time.strftime("%Y-%m-%d-%H-%M-%S")

    os.makedirs(f"nav2d-dataset-{date_time}", exist_ok=True)
    pickle.dump(demos, open(f"nav2d-dataset-{date_time}/{1000}demos.pkl".format(date_time), "wb"))
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_demo', type=int, default=1000)
    args = parser.parse_args()
    
    collect_demo_dataset(args.num_demo)