from os import getcwd, mkdir
from os.path import join, isdir
from datetime import datetime

from lib import map_gen as mg

class gridMazeGen:
    def __init__(self, n_maze, dim, target_folder_name, r_default=-1, r_hitwall=-10, seed=None):
        # Get current date
        self.now = datetime.now()
        self.timestamp = self.now.strftime('%y%m%d')

        # Generate Maze
        self.n_maze = n_maze
        self.r_default = r_default
        self.r_hitwall = r_hitwall
        print(f"Generating {n_maze} maze(s) at {self.now.strftime('%Y/%m/%d-%H:%M:%S')}")
        self.mazes = [mg.Maze(dim, seed) for _ in range(n_maze)]
        for maze in self.mazes:
            maze.make_maze()

        self.current_dir = getcwd()
        self.results_folder_name = target_folder_name
        self.results_dir = self.check_dir(join(self.current_dir, self.results_folder_name))

    def check_dir(self, dir):
        if isdir(dir):
            print(f"{dir} exist. Updating directory.")
        else:
            mkdir(dir)
            print(f"{dir} doesn't exist. Creating  directory.")
        return dir

    def generate_ns(self):
        for maze in self.mazes:
            idx = self.mazes.index(maze)
            print(f'Generating State Transition Matrix for {self.timestamp}_{maze.nx:02}X{maze.ny:02}_{idx}')
            maze.gen_next_state()

    def generate_rt(self):
        for maze in self.mazes:
            idx = self.mazes.index(maze)
            print(f'Generating Reward Matrix for {self.timestamp}_{maze.nx:02}X{maze.ny:02}_{idx}')
            maze.gen_rewards(self.r_default, self.r_hitwall)
    
    def generate_maze_svg(self, maze_config, idx, target_dir, mode, tab_str):
        filename = f'{self.timestamp}{maze_config.nx:02}X{maze_config.ny:02}_{mode}{idx}.svg'
        f = join(target_dir, filename)
        maze_config.write_svg(f, mode)
        print(f'{tab_str}Created {filename}')

    def generate_config_txt(self, maze_config, idx, target_dir, tab_str):
        filename = f'{self.timestamp}{maze_config.nx:02}X{maze_config.ny:02}c{idx}.txt'
        fname = join(target_dir, filename)
        with open(fname, 'w') as f:
            rt = maze_config.reward_matrix
            ns = maze_config.state_transition_matrix
            ### Write Number of state and action
            print(f'{maze_config.nx}', file = f)
            print(f'{maze_config.ny}', file = f)
            print(f'{maze_config.Z}', file = f)
            ### Write state transition matrix
            for state in range(maze_config.N):
                for act in range (maze_config.Z):
                    print(f'{ns[state][act]};', end ='', file = f)
                print('', file = f)
            ### Write reward matrix
            for state in range(maze_config.N):
                for act in range (maze_config.Z):
                    print(f'{rt[state][act]};', end ='', file = f)  
                print('', file = f)
        f.close()
        print(f'{tab_str}Created {filename}')
    
    def save_results(self):
        for maze in self.mazes:
            idx = self.mazes.index(maze)
            target_folder = f'{self.timestamp}_{maze.nx:02}X{maze.ny:02}_{idx}'
            target_dir = join(self.results_dir, target_folder)
            while isdir(target_dir):
                idx += 1
                target_folder = f'{self.timestamp}_{maze.nx:02}X{maze.ny:02}_{idx}'
                target_dir = join(self.results_dir, target_folder)
            target_dir = self.check_dir(target_dir)

            print(f'In {target_dir}:')
            tab_str = '\t'
            ### Generate Maze SVG with state number
            self.generate_maze_svg(maze, idx, target_dir, "s", tab_str)

            ### Generate Maze SVG with (x,y) coordinate
            self.generate_maze_svg(maze, idx, target_dir, "c", tab_str)

            ### Generate maze config files
            self.generate_config_txt(maze, idx, target_dir, tab_str)
