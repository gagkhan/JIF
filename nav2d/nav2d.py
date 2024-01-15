
import numpy as np
import matplotlib.pyplot as plt



class Map2D:

    def __init__(self) -> None:
        
        self.arena_radius = 1
        self.arena_center = (0, 0)

        self.obstacle_radius = 0.01
        self.obstacle_center_radius = 0.5
        
        self.obstacle_angles = np.array([0, 90, 180, 270, 360])



class RRTExpert:

    def __init__(self, map: Map2D) -> None:
        self.map = map

        self.start = None
        self.goal = None

        self.goal_reached = False
        self.edge_length = 0.05


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
            np.vstack([self.nodes, p_new])
            np.vstack([self.edges, p_new - p_near])
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
        edge = self.edges_length * (p_rand - p_near) / np.linalg.norm(p_rand - p_near)
        p_new = p_near + edge
        return p_new
    

    def _is_valid_node(self, p_new):
        # Check if the new node is valid
        if np.linalg.norm(p_new) > self.map.arena_radius:
            return False
        for angle in self.map.obstacle_angles:
            if np.linalg.norm(p_new - self.map.obstacle_center(angle)) < self.map.obstacle_radius:
                return False
        return True
    

    def _check_goal_reached(self):
        # Check if the goal is reached
        if np.linalg.norm(self.nodes[-1] - self.goal) < self.edge_length:
            self.goal_reached = True

    
    def _get_path(self):
        # Get the path
        path = []
        index = -1
        while index != 0:
            path.append(self.nodes[index])
            index = self.parents[index]
        path.append(self.start)
        path.reverse()
        path = np.array(path)
        path_edges = self.path[1:] - self.path[:-1]
        return path, path_edges
    


def main():

    map = Map2D()
    expert = RRTExpert(map)

    start = np.array([0.5, 0.5])
    goal = np.array([-0.5, -0.5])

    expert.plan(start, goal)

    path, path_edges = expert._get_path()

    print(path)
    print(path_edges)

    # Plot path
    plt.plot(path[:, 0], path[:, 1], color='yellow')
    plt.fill_between(path[:, 0], path[:, 1], color='yellow')

    # Set plot limits and aspect ratio
    plt.xlim(-map.arena_radius, map.arena_radius)
    plt.ylim(-map.arena_radius, map.arena_radius)
    plt.gca().set_aspect('equal', adjustable='box')

    # Display the plot
    plt.show()


if __name__ == '__main__':
    main()
    

