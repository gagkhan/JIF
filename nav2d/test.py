from nav2d import Map2D
from models import OIL
import numpy as np
import torch
import time
import matplotlib.pyplot as plt

class Robot:
    
    def __init__(self, map: Map2D) -> None:
        self.map = map
        self.pos = np.array([0, 0])
        self.goal = np.array([0, 0])

        # Rendering
        self.fig = None
        self.t = 0
        
    def step(self, action):
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
            arena = plt.Circle(self.map.arena_center, self.map.arena_radius, color='green')
            self.ax.add_artist(arena)

            # Plot obstacles
            for angle in self.map.obstacle_angles:
                obstacle_center = self.map.obstacle_center_radius * np.array([ np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))])
                obstacle = plt.Circle(obstacle_center, self.map.obstacle_radius, color='black')
                self.ax.add_artist(obstacle)
            
            self.vis_robot = self.ax.plot(self.pos[0], self.pos[1], 'o', color='red')[0]
            self.vis_goal = self.ax.plot(self.goal[0], self.goal[1], 'o', color='blue')[0]
            
            self.ax.set_xlim([-self.map.arena_radius, self.map.arena_radius])
            self.ax.set_ylim([-self.map.arena_radius, self.map.arena_radius])
            self.ax.set_aspect('equal', adjustable='box')
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
            start = radius[0] * np.array([ np.cos(np.deg2rad(angles[0])), np.sin(np.deg2rad(angles[0]))])
            goal = radius[1] * np.array([ np.cos(np.deg2rad(angles[1])), np.sin(np.deg2rad(angles[1]))])
            if self.map._collision_check(start) and self.map._collision_check(goal):
                self.pos = start
                self.goal = goal
                break
        obs = np.concatenate([self.pos, self.goal])
        self.t = 0
        return obs

    
     
def test():

    print('Loading model...')
    
    model = OIL(obs_dim=4, act_dim=2, hidden_dim=64, num_hidden=2)
    
    model.load_state_dict(torch.load('oil.pt', map_location='cpu'))
    model.eval()

    print('Model loaded.')

    map = Map2D()
    robot = Robot(map)

    obs = robot.reset()

    for i in range(1000):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device='cpu')
        obs_tensor = obs_tensor.unsqueeze(0)
        next_obs_pred, action = model(obs_tensor)
        action = action.squeeze(0)
        action = action.detach().cpu().numpy()
        print(obs)
        print(action)
        obs = robot.step(action)
        print(obs)
        # print(action)

        if np.linalg.norm(obs[:2] - obs[2:]) < 0.1:
            print('Goal reached!')


    
if __name__ == '__main__':
    test()