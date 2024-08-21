import random
# Create a maze using the depth-first algorithm described at
# https://scipython.com/blog/making-a-maze/
# Christian Hill, April 2017.

# Added Modification:
# - SVG drawing also draw maze state and border
class Cell:
    """A cell in the maze.

    A maze "Cell" is a point in the grid which may be surrounded by walls to
    the north, east, south or west.

    """

    # A wall separates a pair of cells in the N-S or W-E directions.
    wall_pairs = {'N': 'S', 'S': 'N', 'E': 'W', 'W': 'E'}

    def __init__(self, x, y):
        """Initialize the cell at (x,y). At first it is surrounded by walls."""

        self.x, self.y = x, y
        self.walls = {'N': True, 'S': True, 'E': True, 'W': True}

    def has_all_walls(self):
        """Does this cell still have all its walls?"""

        return all(self.walls.values())

    def knock_down_wall(self, other, wall):
        """Knock down the wall between cells self and other."""

        self.walls[wall] = False
        other.walls[Cell.wall_pairs[wall]] = False

class Maze:
    """A Maze, represented as a grid of cells."""

    def __init__(self, dim, seed):
        """Initialize the maze grid.
        The maze consists of nx x ny cells and will be constructed starting
        at the cell indexed at (ix, iy).

        """
        
        self.seed = seed
        
        
        # Choose starting point
        self.ix = 0
        self.iy = 0
        
        # Maze dimensions
        self.nx = dim
        self.ny = dim

        # Maze entry point

        self.maze_map = [[Cell(x, y) for y in range(self.ny)] for x in range(self.nx)]
        
        # Number of actions
        self.Z = 4
        
        # Number of states
        self.N = self.nx*self.ny

    def cell_at(self, x, y):
        """Return the Cell object at (x,y)."""

        return self.maze_map[x][y]

    def __str__(self):
        """Return a (crude) string representation of the maze."""

        maze_rows = ['-' * self.nx * 2]
        for y in range(self.ny):
            maze_row = ['|']
            for x in range(self.nx):
                if self.maze_map[x][y].walls['E']:
                    maze_row.append(' |')
                else:
                    maze_row.append('  ')
            maze_rows.append(''.join(maze_row))
            maze_row = ['|']
            for x in range(self.nx):
                if self.maze_map[x][y].walls['S']:
                    maze_row.append('-+')
                else:
                    maze_row.append(' +')
            maze_rows.append(''.join(maze_row))
        return '\n'.join(maze_rows)

    def gen_next_state(self):
        n_states = self.nx * self.ny
        next_states = [[0] * self.Z for i in range(n_states)]

        eastmost = self.nx-1
        southmost = self.ny-1
        for y in range(self.ny):
            for x in range(self.nx):
                down, right, left, up = 0, 1, 2, 3
                agent_pos = x+self.nx*y
                # north_of_pos = agent_pos-self.nx
                south_of_pos = agent_pos+self.nx
                # west_of_pos = agent_pos-1
                east_of_pos = agent_pos+1
                
                # Check if wall exist south of the current position (x,y)
                # Also, check if wall exist norht of (x,y+1)
                if self.cell_at(x, y).walls['S']:
                    # Wall exist, agent stays
                    next_states[agent_pos][down] = agent_pos
                    if (y < southmost):
                        # If (x,y) has south wall, then (x,y+1) has north wall
                        next_states[south_of_pos][up] = south_of_pos
                else:
                    # Wall dont exist, agent can move to the south
                    next_states[agent_pos][down] = south_of_pos
                    if (y < southmost):
                        # If (x,y) doesnt have south wall, then (x,y+1) doenst have north wall.
                        # Therefore, in (x,y+1), agent can move to the north
                        next_states[south_of_pos][up] = agent_pos
                
                # Check if wall exist east of the current position (x,y)
                # Also, check if wall exist west of (x+1,y)       
                if self.cell_at(x, y).walls['E']:
                    # Wall exist, agent stays
                    next_states[agent_pos][right] = agent_pos
                    if (x < eastmost):
                        # If (x,y) has south wall, then (x,y+1) has north wall
                        next_states[east_of_pos][left] = east_of_pos
                else:
                    # Wall dont exist, agent can move to the east
                    next_states[agent_pos][right] = east_of_pos
                    if (x < eastmost):
                        # If (x,y) doesnt have east wall, then (x,y+1) doenst have west wall.
                        # Therefore, in (x,y+1), agent can move to the west 
                        next_states[east_of_pos][left] = agent_pos

                # Special Case 1: Agent on the norhtmost positions of maze
                if (y == 0): 
                    next_states[agent_pos][up] = agent_pos

                # Special Case 1: Agent on the westmost positions of maze
                if (x == 0):
                    next_states[agent_pos][left] = agent_pos

        self.state_transition_matrix = next_states
        return next_states
    
    def gen_rewards(self, r_default, r_wall):
        n_states = self.nx * self.ny
        rewards = [[r_default] * self.Z for i in range(n_states)]
        
        down, right, left, up = 0, 1, 2, 3
        
        eastmost = self.nx-1
        southmost = self.ny-1

        # Add wall rewards
        for y in range(self.ny):
            for x in range(self.nx):
                agent_pos = x+self.nx*y
                north_of_pos = agent_pos-self.nx
                south_of_pos = agent_pos+self.nx
                west_of_pos = agent_pos-1
                east_of_pos = agent_pos+1

                # Check if wall exist south of the current position (x,y)
                # Also, check if wall exist norht of (x,y+1)
                if self.cell_at(x, y).walls['S']:
                    # Wall exist. If agent move down, then it get punishment
                    rewards[agent_pos][down] = r_wall
                    if (y < southmost):
                        # If (x,y) has south wall, then (x,y+1) has north wall
                        rewards[south_of_pos][up] = r_wall
                
                # Check if wall exist east of the current position (x,y)
                # Also, check if wall exist west of (x+1,y)       
                if self.cell_at(x, y).walls['E']:
                    # Wall exist, agent stays
                    rewards[agent_pos][right] = r_wall
                    if (x < eastmost):
                        # If (x,y) has south wall, then (x,y+1) has north wall
                        rewards[east_of_pos][left] = r_wall

                # Special Case 1: Agent on the norhtmost positions of maze
                if (y == 0): 
                    rewards[agent_pos][up] = r_wall

                # Special Case 1: Agent on the westmost positions of maze
                if (x == 0):
                    rewards[agent_pos][left] = r_wall

        self.reward_matrix = rewards
        return rewards
    
    def write_svg(self, filename, svg_set):
        """Write an SVG image of the maze to filename."""

        aspect_ratio = self.nx / self.ny
        # Pad the maze all around by this amount.
        padding = 10
        # Height and width of the maze image (excluding padding), in pixels
        height = 1000
        width = int(height * aspect_ratio)
        # Scaling factors mapping maze coordinates to image coordinates
        scy, scx = height / self.ny, width / self.nx
        # Font size for texts
        font_size = scy/5

        def write_wall(ww_f, ww_x1, ww_y1, ww_x2, ww_y2):
            """Write a single wall to the SVG image file handle f."""

            print('<line x1="{}" y1="{}" x2="{}" y2="{}"/>'.format(ww_x1, ww_y1, ww_x2, ww_y2), file=ww_f)

        def write_coords(wc_f, wc_x, wc_y, wc_tx, wc_ty, wc_set):
            """Write a state coordinate to the SVG image file handle f."""
            if (wc_set=="s"):
                state_number = wc_tx+(wc_ty*self.nx)
                print('<text x="{}" y="{}" class="small">S{}</text>'.format(wc_x, wc_y, state_number), file=wc_f)
            else:
                print('<text x="{}" y="{}" class="small">({},{})</text>'.format(wc_x, wc_y, wc_tx, wc_ty), file=wc_f)

        # Write the SVG image file for maze
        with open(filename, 'w') as f:
            # SVG preamble and styles.
            print('<?xml version="1.0" encoding="utf-8"?>', file=f)
            print('<svg xmlns="http://www.w3.org/2000/svg"', file=f)
            print('    xmlns:xlink="http://www.w3.org/1999/xlink"', file=f)
            print('    width="{:d}" height="{:d}" viewBox="{} {} {} {}">'
                  .format(width + 2 * padding, height + 2 * padding,
                          -padding, -padding, width + 2 * padding, height + 2 * padding),
                  file=f)
            print('<defs>\n<style type="text/css"><![CDATA[', file=f)
            print('line {', file=f)
            print('    stroke: #000000;\n    stroke-linecap: square;', file=f)
            print('    stroke-width: 4;\n}', file=f)
            print(']]>', file=f)
            # Add Font Setting
            print('.small {{ font: bold {}px sans-serif;'.format(font_size), file=f)
            print('         fill: lightgray; }}', file=f)
            print('</style>\n\n</defs>', file=f)

            # Draw layout square
            for x in range(self.nx):
                for y in range(self.ny):
                    print(f'<rect x="{x*scx}" y="{y*scy}" width="{scx}" height="{scy}" fill="none" stroke="gray" stroke-width="1"/>', file = f)
            print('',file=f)

            # Draw State Coordinates
            for x in range(self.nx):
                for y in range(self.ny):
                    wx, wy = (x+0.1)*scx, (y+0.3)*scy
                    write_coords(f, wx, wy, x, y, svg_set)
            print('',file=f)

            # Draw the "South" and "East" walls of each cell, if present (these
            # are the "North" and "West" walls of a neighbouring cell in
            # general, of course).
            for x in range(self.nx):
                for y in range(self.ny):
                    if self.cell_at(x, y).walls['S']:
                        x1, y1, x2, y2 = x * scx, (y + 1) * scy, (x + 1) * scx, (y + 1) * scy
                        write_wall(f, x1, y1, x2, y2)

                    if self.cell_at(x, y).walls['E']:
                        x1, y1, x2, y2 = (x + 1) * scx, y * scy, (x + 1) * scx, (y + 1) * scy
                        write_wall(f, x1, y1, x2, y2)
            print('',file=f)
            # Draw the North and West maze border, which won't have been drawn
            # by the procedure above.
            print('<line x1="0" y1="0" x2="{}" y2="0"/>'.format(width), file=f)
            print('<line x1="0" y1="0" x2="0" y2="{}"/>'.format(height), file=f)
            print('</svg>', file=f)

    def find_valid_neighbours(self, cell):
        """Return a list of unvisited neighbours to cell."""

        delta = [('W', (-1, 0)),
                 ('E', (1, 0)),
                 ('S', (0, 1)),
                 ('N', (0, -1))]
        neighbours = []
        for direction, (dx, dy) in delta:
            x2, y2 = cell.x + dx, cell.y + dy
            if (0 <= x2 < self.nx) and (0 <= y2 < self.ny):
                neighbour = self.cell_at(x2, y2)
                if neighbour.has_all_walls():
                    neighbours.append((direction, neighbour))
        return neighbours

    def make_maze(self):
        if self.seed is not None:
            random.seed(self.seed)
        
        # Choose starting point
        self.ix = random.randint(0, self.nx-1)
        self.iy = random.randint(0, self.ny-1)
        
        # Total number of cells.
        n = self.nx * self.ny
        cell_stack = []
        current_cell = self.cell_at(self.ix, self.iy)
        # Total number of visited cells during maze construction.
        nv = 1

        while nv < n:
            neighbours = self.find_valid_neighbours(current_cell)
            if not neighbours:
                # We've reached a dead end: backtrack.
                current_cell = cell_stack.pop()
                continue

            # Choose a random neighbouring cell and move to it.
            direction, next_cell = random.choice(neighbours)
            current_cell.knock_down_wall(next_cell, direction)
            cell_stack.append(current_cell)
            current_cell = next_cell
            nv += 1