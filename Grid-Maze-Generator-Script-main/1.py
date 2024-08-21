import gymnasium as gym
from gymnasium import spaces
import numpy as np

class CustomMazeEnv(gym.Env):
    def __init__(self, max_steps_per_episode=400, maze_data=None, maze_size=10):
        super(CustomMazeEnv, self).__init__()
        
        # Define the size of the maze
        self.maze_size = maze_size
        self.max_steps_per_episode = max_steps_per_episode
            
        # Define action and observation space
        self.action_space = spaces.Discrete(4)  # [0: up, 1: right, 2: down, 3: left]
        self.observation_space = spaces.Box(low=0, high=127, shape=(self.maze_size, self.maze_size), dtype=np.uint8)

        self.maze = None

        # Initial state
        self.state = (0, 0)
        self.start = (0, 0)
        self.goal = (self.maze_size - 1, self.maze_size - 1)
        
        self.visit_counts = np.zeros((self.maze_size, self.maze_size), dtype=np.int32)
        self.current_step = 0
        self.initialize_maze(maze_data)
        
    def initialize_maze(self, input_string):
        maze_size = self.maze_size
        rows = input_string.strip().split('\n')

        top_wall = np.zeros((maze_size, maze_size), dtype=np.uint8)
        right_wall = np.zeros((maze_size, maze_size), dtype=np.uint8)
        bottom_wall = np.zeros((maze_size, maze_size), dtype=np.uint8)
        left_wall = np.zeros((maze_size, maze_size), dtype=np.uint8)

        parsed_data = []
        for row in rows:
            parsed_data.append([int(x) for x in row.split(';') if x != ''])

        for i in range(maze_size):
            for j in range(maze_size):
                state_index = i * maze_size + j
                actions = parsed_data[state_index]
                bottom_wall[i, j] = 1 if actions[0] == -10 else 0
                right_wall[i, j] = 1 if actions[1] == -10 else 0
                left_wall[i, j] = 1 if actions[2] == -10 else 0
                top_wall[i, j] = 1 if actions[3] == -10 else 0

        self.maze = np.stack([top_wall, right_wall, bottom_wall, left_wall], axis=0)  # Stack along axis 0
        
    def _get_obs(self):
        return self.maze

    def reset(self, seed=None, options=None):
        # Reset the state of the environment to an initial state
        super().reset(seed=seed)
        self.current_step = 0
        self.state = self.start
        self.visit_counts = np.zeros((self.maze_size, self.maze_size), dtype=np.int32)
        self.visit_counts[self.state] += 1
        return self._get_obs(), {}

    def step(self, action):
        self.current_step += 1
        x, y = self.state
        new_x, new_y = x, y

        # Define possible moves
        if action == 0:  # up
            new_x = max(0, x - 1)
        elif action == 1:  # right
            new_y = min(self.maze_size - 1, y + 1)
        elif action == 2:  # down
            new_x = min(self.maze_size - 1, x + 1)
        elif action == 3:  # left
            new_y = max(0, y - 1)

        # Check for walls or boundaries
        wall_hit = False
        if action == 0 and (x == 0 or self.maze[0, x, y] == 1):  # Top wall or top edge
            wall_hit = True
        elif action == 1 and (y == self.maze_size - 1 or self.maze[1, x, y] == 1):  # Right wall or right edge
            wall_hit = True
        elif action == 2 and (x == self.maze_size - 1 or self.maze[2, x, y] == 1):  # Bottom wall or bottom edge
            wall_hit = True
        elif action == 3 and (y == 0 or self.maze[3, x, y] == 1):  # Left wall or left edge
            wall_hit = True
        
        # Determine reward and update state
        if wall_hit:
            reward = -10
        else:
            self.state = (new_x, new_y)
            if self.state == self.goal:
                reward = 10
            else:
                reward = -1  # Default step cost
        
        self.visit_counts[self.state] += 1

        # Check if goal is reached or max steps are reached
        done = False
        is_success = False
        if self.state == self.goal:
            done = True
            is_success = True
        elif self.current_step >= self.max_steps_per_episode:
            done = True
            is_success = False
        
        return self._get_obs(), reward, done, False, {"is_success": is_success}

    
    def render(self, mode='human'):
        maze_render = np.full((self.maze_size * 2 + 1, self.maze_size * 2 + 1), ' ', dtype=str)
        
        # Draw walls
        for i in range(self.maze_size):
            for j in range(self.maze_size):
                cell_y, cell_x = i * 2 + 1, j * 2 + 1
                if self.maze[0, i, j]:  # Top wall
                    maze_render[cell_y - 1, cell_x] = '─'
                if self.maze[1, i, j]:  # Right wall
                    maze_render[cell_y, cell_x + 1] = '│'
                if self.maze[2, i, j]:  # Bottom wall
                    maze_render[cell_y + 1, cell_x] = '─'
                if self.maze[3, i, j]:  # Left wall
                    maze_render[cell_y, cell_x - 1] = '│'

        # Mark start, goal, and current position
        start_y, start_x = 1, 1
        goal_y, goal_x = self.maze_size * 2 - 1, self.maze_size * 2 - 1
        agent_y, agent_x = self.state[0] * 2 + 1, self.state[1] * 2 + 1

        maze_render[start_y, start_x] = 'S'
        maze_render[goal_y, goal_x] = 'G'
        maze_render[agent_y, agent_x] = 'A'

        # Print the maze
        for row in maze_render:
            print(''.join(row))
        print("\n")