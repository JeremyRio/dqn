from datetime import datetime
from rig.type_casts import float_to_fp, fp_to_float, NumpyFloatToFixConverter, NumpyFixToFloatConverter
from os import getcwd, mkdir, listdir
from os.path import join, isdir, splitext
from IPython.display import display, HTML
from base64 import b64encode

class COEgen:
    def __init__(self, target_folder_name):
        self.current_dir = getcwd()
        self.results_folder_name = target_folder_name

    def gen_path(self, c_dir, subdir_name):
        # Create path to sub directory
        subdir_path = join(c_dir, subdir_name)

        # Check if subdirectory exist. If doesn't exist generate new one.
        if isdir(subdir_path)==False:
            print("Folder '%s' does not exist. " % subdir_name, end = "")
            mkdir(subdir_path)
            print("Created a new folder named '%s'."% subdir_name)
        else:
            print("Folder named '%s' exist." % subdir_name)
            
        # Return geneated path    
        return subdir_path
    
    def scan_file(self, dir, quiet=False):
        # Get list of maze config available in the folder
        list_files = listdir(dir)

        # Sort file
        list_files.sort()
        
        # # Filter to only bitstream files
        # mazeConfig_list = []
        # for file in maze_files:
        #     extension = os.path.splitext(file)[1]
        #     if (extension == ".txt"):
        #         mazeConfig_list.append(file)

        # Print all files
        if not quiet:
            print("%d maze confing file(s) detected."% len(list_files))
            for file in list_files:
                idx = list_files.index(file) + 1
                print("%d) %s"% (idx, file))
        
        return list_files
    
    def select_file(self, dir):
        file_list = self.scan_file(dir)
        # Get user input to select overlay from overlay file list
        loop = True
        while(loop):
            idx = int(input('\nInput Select Index (1-%d) : '%(len(file_list))))-1
            if ((0 <= idx) and (idx < len(file_list))):
                loop = False
                selected_file = file_list[idx]
                print("\nSelected '%s'"% selected_file)
            else:
                print('Input out of range. Please try again.')
        return selected_file
    
    def choose_config(self):
        results_dir = self.gen_path(self.current_dir, self.results_folder_name)
        target_file = self.select_file(results_dir)
        target_dir = self.gen_path(results_dir, target_file)
        
        list_of_file = self.scan_file(target_dir)

        # Filter to only bitstream files
        config_file_list = []
        for file in list_of_file:
            extension = splitext(file)[1]
            if (extension == ".txt"):
                config_file_list.append(file)
        
        if (len(config_file_list)!=1):
            print(f'Mutliple .txt files detected. Selected {config_file_list[0]}')
        else:
            print(f'Selected {config_file_list[0]}')
        config_file = config_file_list[0]

        self.target_dir = target_dir
        self.config_file = config_file
    
    def load_mazeConfig(self):
        # Read Maze config file
        config_target = join(self.target_dir, self.config_file)
        with open(config_target, 'r') as f:
            print(f'Loading {self.config_file}...')
            lines = f.readlines()
            total_line = len(lines)
            print(f'\tFile consists of {total_line} lines of data.')
            f.close()
        
        # Load Maze Size
        maze_x = int(lines[0])
        maze_y = int(lines[1])
        total_state = maze_x * maze_y
        print(f'\tMaze size loaded. {maze_x}X{maze_y} ({total_state} states)')

        # Load total action
        total_act = int(lines[2])
        print(f'\tNumber of action loaded. There are {total_act} actions')

        # Load Next State list
        NS_list = [[0] * total_act for i in range(total_state)]
        for i in range(total_state):
            x = lines[3+i].split(';')
            x.remove('\n')
            for j in range(total_act):
                NS_list[i][j] = int(x[j])
        print('\tNext State list loaded.')

        # Load Current Reward List
        RT_list = [[0.0] * total_act for i in range(total_state)]
        for i in range(total_state):
            x = lines[3+total_state+i].split(';')
            x.remove('\n')
            for j in range(total_act):
                RT_list[i][j] = float(x[j])
        print('\tCurrent Reward list loaded.')
        print(f'Finish loading {self.config_file}')

        self.N = total_state
        self.Z = total_act
        self.NS = NS_list
        self.RT = RT_list
    
    def print_NS(self):
        print('NEXT STATE MEMORY')
        for i in range(self.N):
            print(f'S{(i):03d}: ',end='')
            for j in range(self.Z):
                print(f' {(self.NS[i][j]):03d} |',end='')
            print()
    
    def print_RT(self):
        print('REWARD MEMORY')
        for i in range(self.N):
            print(f'S{(i):03d}: ',end='')
            for j in range(self.Z):
                val = self.RT[i][j]
                if (val == -1):
                    print(f'  {(val):.1f} |',end='')
                elif (val == -10):
                    print(f' {(val):.1f} |',end='')
                else:
                    print(f'  {(val):.1f} |',end='')
            print()

    def add_goal_state(self, width, goal_reward=10):
        # Display SVG
        print(f"Selected Maze Map")
        ## Scan for SVG image in config folder
        list_of_file = self.scan_file(self.target_dir, quiet=True)
        SVG_file = []
        for file in list_of_file:
            fname = splitext(file)
            if (fname[1] == ".svg"):
                SVG_file.append(file)
        ## Select and load SVG file into HTML
        svg_file = join(self.target_dir, SVG_file[1])
        with open(svg_file, "rb") as image_file:
            svg_base64 = str(b64encode(image_file.read()),'utf-8')
        html_template = f'<img src="data:image/svg+xml;base64,{svg_base64}" width="{width}"/>'
        display(HTML(html_template))

        # Ask user for the goal_state
        N_ = self.N-1
        loop = True
        while(loop):
            self.goal_state = int(input(f'Goal State (0-{N_}): '))
            if (self.goal_state > (N_)):
                print("Goal State out of bounds. Try again")
            else:
                loop = False
                print(f"Maze Goal is S{self.goal_state:03d}")

        # Updates Current Reward list
        for i in range(self.N):
            for j in range(self.Z):
                if (self.NS[i][j]==self.goal_state)&(i != self.goal_state):
                    self.RT[i][j] = goal_reward
        print('Current Reward list updated.')
    
    def gen_COE(self, dat_width, frac_bit):
        folder_name = f'COE_S{self.N}G{self.goal_state}_Q{dat_width}-{frac_bit}'
        coe_dir = join(self.target_dir, folder_name)

        # Create the directory if it does not exist
        if isdir(coe_dir):
            print("File '% s' already exist. NO COE FILES WILL BE GENERATED." % coe_dir)
        else:
            mkdir(coe_dir)
            print("File '% s' created" % coe_dir)
        
            ## Generate Z COE files for NS_MEM
            print(f"In {coe_dir}:")
            for a in range(self.Z):
                filename = f'S{self.N}_NS{a}_MEM.coe'
                path = join(coe_dir, filename)
                with open(path, 'w') as f:
                    print('memory_initialization_radix=10;', file = f)
                    print('memory_initialization_vector=',  end ='', file = f)
                    for s in range(self.N):
                        print(f' {self.NS[s][a]}',  end ='', file = f)
                    print(';', file = f)
                f.close()
                print(f"\tGenerated {filename}")

            ## Generate a single COE file for RT_MEM
            filename = f'S{self.N}_RT_MEM.coe'
            path = join(coe_dir, filename)
            with open(path, 'w') as f:
                print('memory_initialization_radix=10;', file = f)
                print('memory_initialization_vector=',  end ='', file = f)
                for s in range(self.N):
                    for a in range(self.Z):
                        val0 = self.RT[s][a]*(2**frac_bit)
                        val1 = (int(val0) + (1 << dat_width)) % (1 << dat_width)
                        print(f' {val1}',  end ='', file = f)
                print(';', file = f)
            f.close()
            print(f"\tGenerated {filename}")