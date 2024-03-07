import time

import matplotlib.pyplot as plt
import numpy as np


class Map2D:

    def __init__(self) -> None:

        self.arena_radius = 1.5
        self.arena_center = (0, 0)

        self.obstacle_radius = 0.2
        self.obstacle_center_radius = 0.7

        self.obstacle_angles = np.array([45, 135, -45, -135])

    def _collision_check(self, p):
        # Check if the new node is valid
        if np.linalg.norm(p) > self.arena_radius:
            return False
        for angle in self.obstacle_angles:
            obstacle_center = self.obstacle_center_radius * np.array(
                [np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]
            )
            if np.linalg.norm(p - obstacle_center) < self.obstacle_radius:
                return False
        return True


class RRTExpert:

    def __init__(self, map: Map2D) -> None:
        self.map = map

        self.start = None
        self.goal = None

        self.goal_reached = False
        self.edge_length = 0.1

    def plan(self, start, goal):
        self.start = start
        self.goal = goal

        self.nodes = np.array([self.start])
        self.edges = np.array([0, 0])
        self.parents = [0]

        self.goal_reached = False

        while not self.goal_reached:
            self._extend()
            self._check_goal_reached()

        self._get_path()

    def _extend(self):

        # Sample a random point
        p_rand = self._sample_random_point()

        # Find the nearest node
        p_near = self._find_nearest_node(p_rand)

        # Find the new node
        p_new = self._find_new_node(p_near, p_rand)

        # Check if the new node is valid
        if self._is_valid_node(p_new):
            self.nodes = np.vstack([self.nodes, p_new])
            self.edges = np.vstack([self.edges, p_new - p_near])
            # Note: when a node for p_new is added, the edge associated with it is at the same
            # index as the node.
            self.parents.append(np.argmin(np.linalg.norm(self.nodes - p_near, axis=1)))

    def _sample_random_point(self):
        # Sample a random point
        x = np.random.uniform(-self.map.arena_radius, self.map.arena_radius)
        y = np.random.uniform(-self.map.arena_radius, self.map.arena_radius)
        return np.array([x, y])

    def _find_nearest_node(self, p_rand):
        # Find the nearest node
        distances = np.linalg.norm(self.nodes - p_rand, axis=1)
        p_near = self.nodes[np.argmin(distances)]
        return p_near

    def _find_new_node(self, p_near, p_rand):
        # Find the new node
        edge = self.edge_length * (p_rand - p_near) / np.linalg.norm(p_rand - p_near)
        p_new = p_near + edge
        return p_new

    def _is_valid_node(self, p_new):
        return self.map._collision_check(p_new)

    def _check_goal_reached(self):
        # Check if the goal is reached
        if np.linalg.norm(self.nodes[-1] - self.goal) < self.edge_length:
            self.goal_reached = True

    def _get_path(self):
        # Get the path
        path = []
        edges = []
        index = -1
        while index != 0:
            path.append(self.nodes[index])
            # In our data structure, the edge associated with a node is at the
            # same index as the node. So, we can use the index to get the edge
            # associated with the node as done below.
            edges.append(self.edges[index])
            index = self.parents[index]

        path.append(self.start)
        path.reverse()
        path = np.array(path)
        actions = path[1:] - path[:-1]
        return path, actions


def single_demo_test():

    map = Map2D()
    expert = RRTExpert(map)

    start = np.array([0.5, 0.0])
    goal = np.array([-0.5, 0.0])

    # random start and goal
    while True:
        start = np.random.uniform(-map.arena_radius, map.arena_radius, size=2)
        goal = np.random.uniform(-map.arena_radius, map.arena_radius, size=2)
        if expert._is_valid_node(start) and expert._is_valid_node(goal):
            break

    expert.plan(start, goal)

    path, path_edges = expert._get_path()

    print(path)
    print(path_edges)

    # Plot path
    plt.plot(path[:, 0], path[:, 1], color="yellow")
    # plt.fill_between(path[:, 0], path[:, 1], color='yellow')

    # Plot arena
    arena = plt.Circle(map.arena_center, map.arena_radius, color="green")
    plt.gcf().gca().add_artist(arena)

    # Plot obstacles
    for angle in map.obstacle_angles:
        obstacle_center = map.obstacle_center_radius * np.array(
            [np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]
        )
        obstacle = plt.Circle(obstacle_center, map.obstacle_radius, color="black")
        plt.gcf().gca().add_artist(obstacle)

    # Plot start and goal
    plt.plot(start[0], start[1], "o", color="red")
    plt.plot(goal[0], goal[1], "o", color="blue")

    # Set plot limits and aspect ratio
    plt.xlim(-map.arena_radius, map.arena_radius)
    plt.ylim(-map.arena_radius, map.arena_radius)
    plt.gca().set_aspect("equal", adjustable="box")

    # Display the plot
    plt.show()


def collect_demo_dataset(num_demo=1000):
    map = Map2D()
    expert = RRTExpert(map)

    for i in range(num_demo):
        angles = np.random.uniform(-180, 180, size=2)
        radius = np.random.uniform(0, map.arena_radius, size=2)
        start = radius[0] * np.array([np.cos(np.deg2rad(angles[0])), np.sin(np.deg2rad(angles[0]))])
        goal = radius[1] * np.array([np.cos(np.deg2rad(angles[1])), np.sin(np.deg2rad(angles[1]))])
        if expert._is_valid_node(start) and expert._is_valid_node(goal):
            expert.plan(start, goal)
            path, path_edges = expert._get_path()
            print(path)
            print(path_edges)
            print("----------------------")


class Robot:
    """2D robot simulator"""

    def __init__(self, map: Map2D) -> None:
        self.map = map

        # Initial position and goal position of the robot
        # Call reset() to initialize
        self.pos = np.array([0, 0])
        self.goal = np.array([0, 0])

        # Rendering
        self.fig = None
        self.t = 0
        self.reset_called_once = False
        self.render_sleep = 0.05
        self.auto_reset = True

    def step(self, action):
        # Call reset() first because the robot needs to be initialized
        if self.reset_called_once is False:
            raise Exception("Call reset() first")

        assert action.shape == self.pos.shape, "action shape does not match"
        new_pos = self.pos + action
        if self.map._collision_check(new_pos):
            self.pos = new_pos

        self.t += 1
        self.render()
        if self.auto_reset:
            if np.linalg.norm(self.pos - self.goal) < 0.1 or self.t > 50:
                self.reset()
        return self.pos.copy()

    def render(self):
        # Render the robot and goal position
        if self.fig is None:
            # Create figure and axis for rendering the map.
            # Executed only once and the data is updated in subsequent calls
            self.fig = plt.figure()
            self.ax = self.fig.add_subplot(111)

            # Plot arena
            arena = plt.Circle(self.map.arena_center, self.map.arena_radius, color="snow")
            self.ax.add_artist(arena)

            # Plot obstacles
            for angle in self.map.obstacle_angles:
                obstacle_center = self.map.obstacle_center_radius * np.array(
                    [np.cos(np.deg2rad(angle)), np.sin(np.deg2rad(angle))]
                )
                obstacle = plt.Circle(
                    obstacle_center, self.map.obstacle_radius, color="lightsteelblue"
                )
                self.ax.add_artist(obstacle)

            self.vis_robot = self.ax.plot(
                self.pos[0], self.pos[1], "o", color="dodgerblue", markersize="12"
            )[0]
            self.vis_goal = self.ax.plot(
                self.goal[0], self.goal[1], "*", color="crimson", markersize="15"
            )[0]

            self.ax.set_xlim([-self.map.arena_radius, self.map.arena_radius])
            self.ax.set_ylim([-self.map.arena_radius, self.map.arena_radius])
            self.ax.set_aspect("equal", adjustable="box")
            self.ax.set_axis_off()
            plt.show(block=False)

        self.vis_robot.set_data(self.pos)
        self.vis_goal.set_data(self.goal)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        time.sleep(self.render_sleep)

    def set_render_sleep(self, sleep_time):
        self.render_sleep = sleep_time

    def reset(self):
        self.reset_called_once = True
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
        self.t = 0
        return self.pos.copy(), goal.copy()


if __name__ == "__main__":

    single_demo_test()
