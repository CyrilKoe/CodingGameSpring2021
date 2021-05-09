import random

TOPO = []

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
    map_size = sum([6*i for i in range(diameter)])+1
    topology = {i : [-1 for i in range(6)] for i in range(map_size)}
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


##########################
# WORLD            #######
##########################

class Cell:
    def __init__(self, index):
        self.index = index

NO_CELL = Cell(-1)

class WorldCell(Cell):
    def __init__(self, index=-1, richness=0, owner=-1, tree=-1, shadow=0, is_dormant=False, neighbors=[NO_CELL for i in range(6)]):
        #super().__init__(index)
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
    def __init__(self, day=0, nutrients=0, cells=[], trees=[[],[]], suns=[0, 0], points=[0, 0], topology=[]):
        self.day = day
        self.sun_direction = self.day % 6
        self.nutrients = nutrients
        self.cells = cells.copy()
        self.trees = [trees[0].copy(), trees[1].copy()]
        self.suns = suns.copy()
        self.points = points.copy()
        self.topology = topology
    
    def copy(self):
        return State(day=self.day, nutrients=self.nutrients, cells=[cell.copy() for cell in self.cells], trees=self.trees.copy(), suns=self.suns.copy(), points=self.points.copy(), topology=self.topology)
    
    def tree_cost(self,player, size):
        res = 0
        for cell_tree in self.trees[player]:
            if self.cells[cell_tree].tree == size:
                res += 1
        return res
    
    def plant(self, player, source, destination, cost=-1):
        if self.cells[destination].tree != -1 or self.cells[destination].richness == 0:
            return False
        if cost == -1: # To not redo the calculation
                cost = self.tree_cost(player, 0)
        if self.suns[player] < cost:
            return False
        self.cells[destination].plant(player)
        self.trees[player].append(destination)
        self.suns[player] -= cost
        self.cells[source].is_dormant = True
        self.cells[destination].is_dormant = True
        return True
    
    def grow(self, player, index, my_tree_cost=-1):
        size_before = self.cells[index].tree
        if size_before == -1:
            return False
        first_cost = 1 if size_before == 1 else (3 if size_before == 2 else 7)
        cost = first_cost + (self.tree_cost(player, size_before+1) if my_tree_cost == -1 else my_tree_cost)
        print(player, index, cost, size_before)
        if cost > self.suns[player]:
            return False
        self.suns[player] -= cost
        self.cells[index].is_dormant = True
        return True
    
    def update_shadows(self, sun_direction):
        for cell in self.cells:
            cell.shadow_size = 0
        for cell in self.cells:
            new_shadow_size = min(0, cell.tree)
            for i in range(new_shadow_size):
                cell = cell.neighbors[sun_direction]
                if cell.index == -1:
                    break
                cell.shadow_size = new_shadow_size
    
    def next_day(self):
        self.day += 1
        self.sun_direction = self.day % 6
        self.update_shadows(self.sun_direction)
        for i in range(len(self.suns)):
            for tree in self.trees[i]:
                self.suns[i] += self.cells[tree].tree
        self.wake_all()
        return self
    
    def wake_all(self):
        for player_trees in self.trees:
            for tree_cell in player_trees:
                self.cells[tree_cell].is_dormant = False

    def nodes_around(self, root, max_distance):
        res = []
        to_visit = [(root, 0)]
        visited = []
        while len(to_visit) > 0:
            node, distance = to_visit[0]
            visited += [node]
            to_visit = to_visit[1:]

            res.append(node)
            for neighbor in self.cells[node].neighbors:
                if neighbor != -1 and (neighbor not in visited) and (neighbor not in [to_vis[0] for to_vis in to_visit]) and distance < max_distance:
                    to_visit += [(neighbor.index, distance+1)]
        return res



class Game:
    def __init__(self, player_0, player_1, diameter=4, first_trees=2, first_stones=2):
        self.topology = build_map(diameter)

        # Richness
        self.richness = {}
        nodes_distances = nodes_around(self.topology, 0, diameter-1)
        for distance in nodes_distances:
            nodes_distances[distance].sort()
            for node in nodes_distances[distance]:
                self.richness[node] = diameter-distance if distance > 1 else 3
        # Cells
        cells = []
        for cell_idx in self.topology:
            cells += [WorldCell(index=cell_idx, richness=self.richness[cell_idx])]
        for cell in cells:
            for direction, neighbor_idx in enumerate(self.topology[cell.index]):
                if self.topology[cell.index][direction] != -1:
                    cell.add_neighbor(cells[neighbor_idx], direction)
        # Firsts trees and stones
        to_plant = first_trees
        to_stone = first_stones
        trees = [[], []]
        while to_plant+to_stone > 0:
            dist = random.randint(1, diameter-1)
            n = len(nodes_distances[dist])
            idx = random.randint(0, n-1)
            cell_idx_1 = nodes_distances[dist][idx]
            cell_idx_2 = nodes_distances[dist][int(idx+n/2)%n]
            if cells[cell_idx_1].owner != -1 or cells[cell_idx_1].richness == 0:
                continue
            if to_plant > 0:
                cells[cell_idx_1].plant(0)
                cells[cell_idx_1].tree = 1
                cells[cell_idx_2].plant(1)
                cells[cell_idx_2].tree = 1
                to_plant -= 1
                trees[0].append(cell_idx_1)
                trees[1].append(cell_idx_2)
            elif to_stone > 0:
                cells[cell_idx_1].richness = 0
                cells[cell_idx_2].richness = 0
                to_stone -= 1   
        self.state = State(day=0, nutrients=20, cells=cells, trees=trees, topology=self.topology)
        self.player_0 = player_0
        self.player_1 = player_1 

    def get_copy_state(self):
        return self.state.copy()
    
    def play_turn(self):
        actions_0 = self.player_0.play(self.get_copy_state())
        actions_1 = self.player_1.play(self.get_copy_state())
        print(actions_0)
        print(actions_1)
        self.state.next_day()


##########################
# PLAYER           #######
##########################


class Player:
    def __init__(self, index):
        self.index = index
    
    def play(self, state):
        actions = ""
        return actions

class IA_1(Player):
    def __init__(self, index):
        super().__init__(index)
    
    def play(self, state):
        possibles_actions = []
        me = self.index
        for tree in state.trees[me]:
            for cell in state.nodes_around(tree, state.cells[tree].tree):
                try_state = state.copy()
                if try_state.plant(me, tree, cell):
                    possibles_actions.append(('GROW '+str(cell), try_state.next_day().suns[me]))
        print(possibles_actions)
        return ""


##########################
# MAIN             #######
##########################


def main():
    player_0 = IA_1(0)
    player_1 = IA_1(1)
    game = Game(player_0, player_1, diameter = 4, first_trees = 2,  first_stones = 2)
    game.play_turn()
    

if __name__ == "__main__":
    main()
