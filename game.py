import random

MAP_SIZE = 37
DIAMETER = 4

##########################
# WORLD GENERATION #######
##########################

def add_neighbor(topology, node, neighbor, direction):
    topology[node][direction%6] = neighbor
    topology[neighbor][(direction+3)%6] = node
    return

def connect_circles(topology, diameter):
    node = 1
    for circle in range(1, diameter):
        direction = 2
        for j in range(6*circle):
            if j == 6*circle-1:
                add_neighbor(topology, node, node-j, 1)
            else:
                add_neighbor(topology, node, node+1, direction)
                if (j+1)%circle == 0:
                    direction += 1
            node += 1
    return

def connect_up_side(topology, node, circle, direction):
    perimeter = circle*6
    for i in range(circle):
        neighbor = node + perimeter + direction
        add_neighbor(topology, node, neighbor, 0+direction)
        up_neighbor = topology[neighbor][(2+direction)%6]
        add_neighbor(topology, node, up_neighbor, 1+direction)
        if i == 0:
            down_neighbor = topology[neighbor][(4+direction)%6]
            add_neighbor(topology, node, down_neighbor, 5+direction)
        node += 1
    return node

def build_map(diameter=4):
    topology = {i : [-1 for i in range(6)] for i in range(MAP_SIZE)}
    connect_circles(topology, diameter)
    for i in range(0, 6):
        add_neighbor(topology, 0, i+1, i)
    node = 1
    for circle in range(1,diameter):
        for direction in range(6):
            if circle < diameter-1:
                node = connect_up_side(topology, node, circle, direction)
    return topology

##########################
# USEFUL FUNCTIONS #######
##########################

def nodes_around(topology, root, max_distance):
    res = {i : [] for i in range(max_distance+1)}
    to_visit = [(root, 0)]
    visited = []
    while len(to_visit) > 0:
        node, distance = to_visit[0]
        visited += [node]
        to_visit = to_visit[1:]

        res[distance].append(node)
        for neighbor in topology[node]:
            if neighbor != -1 and (neighbor not in visited) and (neighbor not in [to_vis[0] for to_vis in to_visit]) and distance < max_distance:
                to_visit += [(neighbor, distance+1)]
    return res

def update_shadows(cells, sun_direction):
    for cell in cells:
        cell.shadow_size = 0
    for cell in cells:
        new_shadow_size = min(0, cell.tree)
        for i in range(new_shadow_size):
            cell = cell.neighbors[sun_direction]
            if cell.index == -1:
                break
            cell.shadow_size = new_shadow_size
            


##########################
# WORLD            #######
##########################

class Cell:
    def __init__(self, index):
        self.index = index

NO_CELL = Cell(-1)

class WorldCell(Cell):
    def __init__(self, index=-1, richness=0, owner=-1, tree=-1, shadow=0, is_dormant=False, neighbors=[NO_CELL for i in range(6)]):
        self.index = index
        self.richness = richness
        self.owner = owner
        self.tree = tree
        self.shadow = shadow
        self.is_dormant = is_dormant
        self.neighbors = neighbors.copy()
    
    def add_neighbor(self, neighbor_cell, direction):
        self.neighbors[direction] = neighbor_cell
        return
    
    def plant(self, player):
        if self.tree != -1 or self.owner != -1:
            assert(False)
        if self.richness == 0:
            assert(False)
        self.tree = 0
        self.owner = player
        return
    
    def grow(self, player):
        if self.tree == -1 or self.owner != player:
            assert(False)
        if self.tree == 3:
            assert(False)
        self.tree += 1
        return
    
    def copy(self):
        return WorldCell(self.index, self.richness, self.owner, self.tree, self.shadow, self.is_dormant, self.neighbors.copy())

class State:
    def __init__(self, day=0, nutrients=0, cells=[], trees=[]):
        self.day = day
        self.sun_direction = self.day % 6
        self.nutrients = nutrients
        self.cells = cells.copy()
        self.trees = []
    
    def copy(self):
        return State(self.day, self.nutrients, [cell.copy() for cell in self.cells], self.trees.copy())
    
    def plant(self, player, index):
        self.cells[index]


class Game:
    def __init__(self):
        self.topology = build_map(diameter=DIAMETER)
        self.players = []
        # Richness
        self.richness = {}
        nodes_distances = nodes_around(self.topology, 0, DIAMETER-1)
        for distance in nodes_distances:
            nodes_distances[distance].sort()
            for node in nodes_distances[distance]:
                self.richness[node] = DIAMETER-distance if distance > 1 else 3
        # Cells
        cells = []
        for cell_idx in self.topology:
            cells += [WorldCell(cell_idx, self.richness[cell_idx])]
        for cell in cells:
            for direction, neighbor_idx in enumerate(self.topology[cell.index]):
                if self.topology[cell.index][direction] != -1:
                    cell.add_neighbor(cells[neighbor_idx], direction)

        # Firsts trees and stones
        to_plant = 2
        to_stone = 2
        while to_plant+to_stone > 0:
            dist = random.randint(1, 3)
            n = len(nodes_distances[dist])
            idx = random.randint(0, n-1)
            cell_idx_1 = nodes_distances[dist][idx]
            cell_idx_2 = nodes_distances[dist][int(idx+n/2)%n]
            if cells[cell_idx_1].owner != -1 or cells[cell_idx_1].richness == 0:
                continue
            if to_plant > 0:
                cells[cell_idx_1].plant(0)
                cells[cell_idx_2].plant(1)
                to_plant -= 1
            elif to_stone > 0:
                cells[cell_idx_1].richness = 0
                cells[cell_idx_2].richness = 0
                to_stone -= 1
    
        self.state = State(0, 20, cells)

    def get_copy_state(self):
        return self.state.copy()
    
    def add_player(self, player):
        self.players.append(player) 
    
    def play_turn(self):
        self.sun_direction = self.day % 6

        self.day += 1




##########################
# PLAYER           #######
##########################


class Player:
    def __init__(self, index):
        self.index = index



##########################
# MAIN             #######
##########################


def main():
    game = Game()

    new_state = game.get_copy_state()
    

if __name__ == "__main__":
    main()
