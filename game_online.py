import sys
import math
import random
import numpy as np

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
        return State(day=self.day, nutrients=self.nutrients, cells=[cell.copy() for cell in self.cells], trees=self.trees.copy(), suns=self.suns.copy(), points=self.points.copy(), topology=self.topology, nutrients_decrease=self.nutrients_decrease)
    
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
        first_cost = [0, 1, 3, 7][size_before]
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
            cell.grow(player)
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
    
    def update_nutrients(self):
        self.nutrients -= self.nutrients_decrease
        self.nutrients_decrease = 0
    
    def next_day(self, num_days = 1):
        for i in range(num_days):
            self.day += 1
            self.sun_direction = self.day % 6
            self.update_shadows(self.sun_direction)
            self.update_nutrients()
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
# PLAYERS          #######
##########################


class Player:
    def __init__(self, index):
        self.index = index
    
    def play(self, state):
        actions = "WAIT"
        return actions



##########################
# IA_1             #######
##########################


class IA_1(Player):
    def __init__(self, index):
        super().__init__(index)

    def play(self, state):
        depth = 10
        last_depth = 0
        evaluated_states = [self.all_actions(state)] + [[] for i in range(depth)]
        for i in range(depth):
            for (actions, last_state, points) in evaluated_states[i]:
                evaluated_states[i+1] += self.all_actions(last_state, actions+' | ')
            if len(evaluated_states[i+1]) > 0:
                last_depth = i+1

        return max(evaluated_states[last_depth], key=lambda tup : tup[2])[0]
    
    def all_actions(self, state, last_actions = ''):
        last_actions_list = []
        if last_actions != '':
            last_actions_list = last_actions.split(' | ')
            last_actions_list = last_actions_list[0:len(last_actions_list)-1]
            if "WAIT" in last_actions_list:
                return []

        me = self.index

        all_trees = [[] for i in range(4)]
        for cell in state.trees[me]:
            all_trees[state.cells[cell].tree].append(cell)

        for i in range(4):
            all_trees[i].sort()
        
        seed_cost = state.tree_cost(me, 0)
        grow_costs = [len(all_trees[i]) for i in range(0,4)]+[666]
        complete_cost = 4

        try_to_seed = []
        try_to_grow = []
        try_to_complete = []

        evaluated_states = [[(last_actions+'WAIT', state.copy(), self.grade_state(state.copy(), 0))], [], [], []]

        for i in range(1, 4):
            n = len(all_trees[i])
            if n > 0:
                tree = all_trees[i][0:n]
                # Try seeding
                for cell_id in state.nodes_around(tree, state.cells[tree].tree):
                    if state.can_plant(me, tree, cell_id, cost=seed_cost)[0]:
                        try_state = state.copy()
                        try_state.plant(me, tree, cell_id, cost=seed_cost)
                        evaluated_states[1].append((last_actions+'SEED '+str(tree)+' '+str(cell_id), try_state, self.grade_state(try_state, 1)))

        for tree in state.trees[me]:
            # Try growing
            if state.can_grow(me, tree, my_tree_cost=grow_costs[state.cells[tree].tree+1])[0]:
                try_state = state.copy()
                try_state.grow(me, tree, my_tree_cost=grow_costs[state.cells[tree].tree+1])
                evaluated_states[2].append((last_actions+'GROW '+str(tree), try_state, self.grade_state(try_state, 2)))
            # Try completing
            if state.can_complete(me, tree)[0]:
                try_state = state.copy()
                try_state.complete(me, tree)
                evaluated_states[3].append((last_actions+'COMPLETE '+str(tree), try_state, self.grade_state(try_state, 3)))
    
        return [max(evaluated_states[k], key=lambda tup : tup[2]) for k in range(len(evaluated_states)) if len(evaluated_states[k]) > 0]

    def grade_state(self, try_state, action): #Action = {0 : WAIT, 1 : SEED, 2 : GROW , 3 : COMPLETE}
        remaining_days = 24-try_state.day
        days_to_test = remaining_days if remaining_days < 5 else 5
        future_state = try_state.copy().next_day(days_to_test) 
        future_suns = future_state.suns[self.index] - future_state.suns[1-self.index] + (5-days_to_test+0.5)*future_state.points[self.index]
        return future_suns


##########################
# IA_2             #######
##########################


class IA_2(Player):
    def __init__(self, index, network=None):
        super().__init__(index)
        self.state_size = 37*3
        std = 0.01
        if network:
            self.network = network.copy()
        else:
            self.network = NeuralNetwork(std, [self.state_size, 30, 1])

    
    def play(self, state):
        me = self.index

        seed_cost = state.tree_cost(me, 0)
        grow_costs = [state.tree_cost(me, i) for i in range(0,4)]+[666]
        complete_cost = 4

        try_to_seed = []
        try_to_grow = []
        try_to_complete = []

        evaluated_states = [('WAIT', self.grade_state(state))]

        for tree in state.trees[me]:
            # Try seeding
            for cell_id in state.nodes_around(tree, state.cells[tree].tree):
                if state.can_plant(me, tree, cell_id, cost=seed_cost)[0]:
                    try_state = state.copy()
                    try_state.plant(me, tree, cell_id, cost=seed_cost)
                    try_to_seed.append(try_state)
                    evaluated_states.append(('SEED '+str(tree)+' '+str(cell_id), self.grade_state(try_state)))
            # Try growing
            if state.can_grow(me, tree, my_tree_cost=grow_costs[state.cells[tree].tree+1])[0]:
                try_state = state.copy()
                try_state.grow(me, tree, my_tree_cost=grow_costs[state.cells[tree].tree+1])
                try_to_grow.append(try_state)
                evaluated_states.append(('GROW '+str(tree), self.grade_state(try_state)))
            # Try completing
            if state.can_complete(me, tree)[0]:
                try_state = state.copy()
                try_state.complete(me, tree)
                try_to_complete.append(try_state)
                evaluated_states.append(('COMPLETE '+str(tree), self.grade_state(try_state)))
        #print(evaluated_states)
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
    
    def copy(self):
        return IA_2(self.index, self.network.copy())
    
    def mutate(self, coverage, std):
        self.network.mutate(coverage, std)


class NeuralNetwork():
    def __init__(self, std, layer_sizes=[], weights = [], biases = []):
        self.weights = []
        self.biases = []
        self.layer_sizes = layer_sizes.copy()
        if len(self.weights) == 0:
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
        return NeuralNetwork(0, self.layer_sizes, self.weights, self.biases)
    
    def mutate(self, coverage, std): # Coverage between 0 and 100
        for i in range(len(self.weights)):
            noise = np.heaviside(np.random.uniform(0, 100, self.weights[i].shape) - coverage, 0) * np.random.randn(self.weights[i].shape[0], self.weights[i].shape[1])
            noise_b = np.heaviside(np.random.uniform(0, 100, self.biases[i].shape) - coverage, 0) * np.random.randn(self.biases[i].shape[0], self.biases[i].shape[1])
            self.weights[i] += noise
            self.biases[i] += noise_b


##########################
# MAIN             #######
##########################

player = IA_1(0)

richnesses = {}
topology = {}

# Read topology and richnesses
number_of_cells = int(input())  # 37
for i in range(number_of_cells):
    index, richness, neigh_0, neigh_1, neigh_2, neigh_3, neigh_4, neigh_5 = [int(j) for j in input().split()]
    richnesses[index] = richness
    topology[index] = [neigh_0, neigh_1, neigh_2, neigh_3, neigh_4, neigh_5]


# Create WorldCells
cells = []
for cell_idx in topology:
    cells += [WorldCell(index=cell_idx, richness=richnesses[cell_idx])]
for cell in cells:
    for direction, neighbor_idx in enumerate(topology[cell.index]):
        if topology[cell.index][direction] != -1:
            cell.add_neighbor(cells[neighbor_idx], direction)


# Create game state
game_state = State(day=0, nutrients=20, cells=cells, topology=topology)


# Game loop
while True:
    day = int(input()) 
    nutrients = int(input())
    sun, score = [int(i) for i in input().split()]
    inputs = input().split()
    opp_sun = int(inputs[0])
    opp_score = int(inputs[1])
    opp_is_waiting = inputs[2] != "0"

    for cell in game_state.cells:
        cell.tree = -1
        cell.owner = -1
        cell.is_dormant = False

    trees = [[], []]
    number_of_trees = int(input())
    for i in range(number_of_trees):
        inputs = input().split()
        cell_index = int(inputs[0])
        size = int(inputs[1])
        is_mine = inputs[2] != "0"
        is_dormant = inputs[3] != "0"

        cell = game_state.get_cell(cell_index)
        cell.tree = size
        cell.owner = 0 if is_mine else 1
        cell.is_dormant = is_dormant
        trees[cell.owner].append(cell_index)
    

    possible_actions = []
    number_of_possible_actions = int(input())  # all legal actions
    for i in range(number_of_possible_actions):
        possible_actions.append(input())   # try printing something from here to start with
    print(day, number_of_possible_actions, possible_actions, file=sys.stderr)
    
    game_state.day = day
    game_state.nutrients = nutrients
    game_state.suns = [sun, opp_sun]
    game_state.score = [score, opp_score]
    game_state.trees = trees
    
    print(player.play(game_state.copy())+' :) ')
