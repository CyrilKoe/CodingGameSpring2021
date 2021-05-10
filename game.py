import random
import numpy as np

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
        super().__init__(index)
        self.richness = richness
        self.owner = owner
        self.tree = tree
        self.shadow = shadow
        self.is_dormant = is_dormant
        self.neighbors = neighbors.copy()
    
    def add_neighbor(self, neighbor_cell, direction):
        self.neighbors[direction] = neighbor_cell
    
    def plant(self, player):
        assert(self.index != -1)
        assert(self.tree == -1)
        assert(self.owner == -1)
        assert(self.richness != 0)
        assert(not self.is_dormant)
        self.tree = 0
        self.owner = player
    
    def grow(self, player):
        assert(self.tree in [0, 1, 2])
        assert(self.owner == player)
        assert(not self.is_dormant)
        self.tree += 1
    
    def copy(self):
        return WorldCell(self.index, self.richness, self.owner, self.tree, self.shadow, self.is_dormant, self.neighbors.copy())


##########################
# STATE            #######
##########################


class State:
    def __init__(self, day=0, nutrients=0, cells=[], trees=[[],[]], suns=[0, 0], points=[0, 0], topology=[], nutrients_decrease=0):
        self.day = day
        self.sun_direction = self.day % 6
        self.nutrients = nutrients
        self.cells = cells.copy()
        self.trees = [trees[0].copy(), trees[1].copy()]
        self.suns = suns.copy()
        self.points = points.copy()
        self.topology = topology
        self.nutrients_decrease = nutrients_decrease
    
    def copy(self):
        return State(day=self.day, nutrients=self.nutrients, cells=[cell.copy() for cell in self.cells], trees=self.trees.copy(), suns=self.suns.copy(), points=self.points.copy(), topology=self.topology)
    
    def tree_cost(self,player, size):
        res = 0
        for cell_tree in self.trees[player]:
            if self.cells[cell_tree].tree == size:
                res += 1
        return res
    
    def can_plant(self, player, source, destination, cost=-1):
        cell = self.get_cell(destination)
        if cell.index == -1:
            return False, -1
        if cell.tree != -1 or cell.richness == 0 or cell.is_dormant:
            return False, -1
        # To not redo calculation
        cost = self.tree_cost(player, 0) if cost == -1 else cost
        if cost > self.suns[player]:
            return False, -1
        return True, cost

    
    def plant(self, player, source, destination, cost=-1):
        cell = self.get_cell(destination)
        can, cost = self.can_plant(player, source, destination, cost)
        if can:
            cell.plant(player)
            self.trees[player].append(destination)
            self.suns[player] -= cost
            self.get_cell(source).is_dormant = True
            cell.is_dormant = True
            return True
        return False
    
    def can_grow(self, player, destination, my_tree_cost=-1):
        cell = self.get_cell(destination)
        if cell.index == -1:
            return False, -1
        if (not cell.tree in [0, 1, 2]) or cell.owner != player or cell.is_dormant:
            return False, -1
        size_before = cell.tree
        first_cost = [1, 3, 7][size_before]
        # To not redo calculation
        cost = first_cost + (self.tree_cost(player, size_before+1) if my_tree_cost == -1 else my_tree_cost )
        if cost > self.suns[player]:
            return False, -1
        return True, cost
    
    def grow(self, player, destination, my_tree_cost=-1):
        cell = self.get_cell(destination)
        can, cost = self.can_grow(player, destination, my_tree_cost)
        if can:
            self.suns[player] -= cost
            cell.is_dormant = True
            cell.tree += 1
            return True
        return False
    
    def can_complete(self, player, tree):
        cell = self.get_cell(tree)
        if cell.index == -1:
            return False, -1
        if (cell.tree != 3) or cell.owner != player:
            return False, -1
        cost = 4
        if cost > self.suns[player]:
            return False, -1
        return True, cost
    
    def complete(self, player, tree):
        cell = self.get_cell(tree)
        can, cost = self.can_complete(player, tree)
        if can:
            self.suns[player] -= cost
            self.points[player] += cell.richness + self.nutrients
            self.nutrients_decrease += 1
            self.cells[tree].owner = -1
            self.cells[tree].size = -1
            self.trees[player].remove(tree)
            return True
        return False
    
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
    
    def next_day(self, num_days = 1):
        for i in range(num_days):
            self.day += 1
            self.sun_direction = self.day % 6
            self.update_shadows(self.sun_direction)
            self.nutrients -= self.nutrients_decrease
            self.nutrients_decrease = 0
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
                if neighbor != -1 and (neighbor.index not in visited) and (neighbor.index not in [to_vis[0] for to_vis in to_visit]) and distance < max_distance:
                    to_visit += [(neighbor.index, distance+1)]
        return res
    
    def get_cell(self, index):
        if index == -1:
            return NO_CELL
        else:
            return self.cells[index]


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
        actions = ""
        possible_actions = {}
        me = self.index

        remaining_days = 24-state.day
        days_to_test = min(10, remaining_days)
        grow_bonus = 5
        max_trees = []
        future_suns_wait = state.copy().next_day(remaining_days).suns[me]

        for tree in state.trees[me]:
            if(state.cells[tree].tree == 3):
                max_trees.append(tree)
                if(state.suns[0] >= 4 and random.randint(0, 100) < 0*(40-15*(state.day-10)**2)):
                    actions = "COMPLETE "+str(tree) +' | ' + actions
                    continue

            try_state = state.copy()
            if try_state.grow(me, tree):
                future_suns = try_state.next_day(days_to_test).suns[me]
                if not future_suns+grow_bonus in possible_actions:
                    possible_actions[future_suns+grow_bonus] = []
                possible_actions[future_suns+grow_bonus].append(('GROW '+str(tree), tree))

            if remaining_days > 4:
                cells_to_try_plant = []
                for cell in state.nodes_around(tree, state.cells[tree].tree):
                    if cell != -1 and state.cells[cell].tree == -1:
                        cells_to_try_plant.append(cell)
                for cell in sorted(cells_to_try_plant):
                    try_state = state.copy()
                    if try_state.plant(me, tree, cell):
                        future_suns = try_state.next_day(days_to_test).suns[me]
                        if not future_suns in possible_actions:
                            possible_actions[future_suns] = []
                        possible_actions[future_suns].append(('SEED '+str(tree)+' '+str(cell), cell))
                        break

        for future_suns in sorted(possible_actions):
            for action in sorted(possible_actions[future_suns], key=lambda tup : tup[1]):
                actions += action[0] + " | "

        if remaining_days <= 2 and len(max_trees) > 0 and state.suns[0] >= 4:
            actions = "COMPLETE " + str(max_trees[0]) + ' | '
        return actions + "WAIT"


class IA_2(Player):
    def __init__(self, index):
        super().__init__(index)
        self.state_size = 37*3
        std = 0.01
        self.network = NeuralNetwork(std, [self.state_size, 30, 1])

    
    def play(self, state):
        me = self.index

        seed_cost = state.tree_cost(me, 0)
        grow_costs = [state.tree_cost(me, i) for i in range(1,4)]
        complete_cost = 4

        try_to_seed = []
        try_to_grow = []
        try_to_complete = []

        evaluated_states = []

        for tree in state.trees[me]:
            # Try seeding
            for cell_id in state.nodes_around(tree, state.cells[tree].tree):
                if state.can_plant(me, tree, cell_id, cost=seed_cost)[0]:
                    try_state = state.copy()
                    try_state.plant(me, tree, cell_id, cost=seed_cost)
                    try_to_seed.append(try_state)
                    evaluated_states.append(('SEED '+str(tree)+' '+str(cell_id), self.grade_state(try_state)))
            # Try growing
            if state.can_grow(me, tree, my_tree_cost=grow_costs[state.cells[tree].tree-1])[0]:
                try_state = state.copy()
                try_state.grow(me, tree, my_tree_cost=grow_costs[state.cells[tree].tree-1])
                try_to_grow.append(try_state)
                evaluated_states.append(('GROW '+str(tree), self.grade_state(try_state)))
            # Try completing
            if state.can_complete(me, tree)[0]:
                try_state = state.copy()
                try_state.complete(me, tree)
                try_to_complete.append(try_state)
                evaluated_states.append(('COMPLETE '+str(tree), self.grade_state(try_state)))
        if len(evaluated_states) == 0:
            return "WAIT"

        evaluated_states.sort(key=lambda tup : tup[1][0], reverse=True)
        return evaluated_states[0][0]
    
    def grade_state(self, try_state):
        state_description = []
        for cell in try_state.cells:
            state_description += [float(cell.is_dormant), float(cell.richness), (cell.tree+1)*(2*float(cell.owner == self.index)-0.5)]
        state_description = np.array(state_description).reshape((self.state_size, 1))
        return self.network.forward(state_description)


class NeuralNetwork():
    def __init__(self, std, layer_sizes=[], weights = [], biases = []):
        self.weights = []
        self.biases = []
        self.layer_sizes = layer_sizes.copy()
        if len(self.weights) != 0:
            for i in range(len(layer_sizes)-1):
                self.weights.append(np.random.randn(layer_sizes[i+1], layer_sizes[i])*std)
                self.biases.append(np.random.randn(layer_sizes[i+1], 1)*std)
        else:
            self.weights = [w.copy() for w in weights]
            self.biases = [b.copy() for b in weights]
    
    def forward(self, input_vector):
        intermediate = input_vector
        for i in range(len(self.weights)):
            intermediate = np.dot(self.weights[i], intermediate)+self.biases[i]
            intermediate = 1/(1 + np.exp(-1*intermediate))
        return intermediate
    
    def copy(self):
        return NeuralNetwork(self, 0, self.layer_sizes, self.weights, self.biases)


##########################
# GAME ENGINE      #######
##########################



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
        actions_0 = self.player_0.play(self.get_copy_state()).split(' | ')
        actions_1 = self.player_1.play(self.get_copy_state()).split(' | ')
        print("Player 0", actions_0)
        print("Player 1", actions_1)
        n_0 = len(actions_0)
        n_1 = len(actions_1)
        for i in range(max(n_0, n_1)):
            action_0 = actions_0[i] if n_0 > i else ''
            action_1 = actions_1[i] if n_1 > i else ''
            self.play_actions([action_0, action_1])
        self.state.next_day()
    
    def play_actions(self, actions = ['', '']):  
        descriptions = [[], []]
        for i in range(len(actions)):
            if actions[i] != '':
                descriptions[i] = actions[i].split(' ')
        
        # Planting at the same spot
        if actions[0] != '' and actions[0][0] == 'S' and actions[0] == actions[1]:
            for action, source, destination in descriptions:
                self.state.get_cell(int(source)).is_dormant = True
                return
        
        for i in range(len(descriptions)):
            if len(descriptions[i]) < 2:
                continue
            if len(descriptions[i]) == 2:
                action, tree = descriptions[i]
                if action[0] == 'G':
                    self.state.grow(i, int(tree))
                    continue
                if action[0] == 'C':
                    self.state.complete(i, int(tree))
                    continue
            if len(descriptions[i]) == 3:
                action, source, destination = descriptions[i]
                self.state.plant(i, int(source), int(destination))
                continue


##########################
# MAIN             #######
##########################


def main():

    player_0 = IA_1(0)
    player_1 = IA_2(1)
    game = Game(player_0, player_1, diameter = 4, first_trees = 2,  first_stones = 2)

    for i in range(24):
        print('')
        print(game.state.day, game.state.suns, game.state.points)
        print([(tree, game.state.cells[tree].tree) for tree in game.state.trees[game.player_0.index]])
        print([(tree, game.state.cells[tree].tree) for tree in game.state.trees[game.player_1.index]])
        game.play_turn()
    

if __name__ == "__main__":
    main()
