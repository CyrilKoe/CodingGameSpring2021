import random
import numpy as np
import sys

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
        depth = 2
        last_depth = 0
        evaluated_states = [self.all_actions(state)] + [[] for i in range(depth)]
        for i in range(depth):
            for (actions, last_state, points) in evaluated_states[i]:
                evaluated_states[i+1] += self.all_actions(last_state, actions+' | ')
            if len(evaluated_states[i+1]) > 0:
                last_depth = i+1
        
        #print(evaluated_states)
        #print(evaluated_states[last_depth])
        #print(max(evaluated_states[last_depth], key=lambda tup : tup[2]))


        return max(evaluated_states[last_depth], key=lambda tup : tup[2])[0]
    
    def all_actions(self, state, last_actions = ''):
        last_actions_list = []
        if last_actions != '':
            last_actions_list = last_actions.split(' | ')
            last_actions_list = last_actions_list[0:len(last_actions_list)-1]
            if "WAIT" in last_actions_list:
                return []


        me = self.index
        seed_cost = state.tree_cost(me, 0)
        grow_costs = [state.tree_cost(me, i) for i in range(0,4)]+[666]
        complete_cost = 4

        try_to_seed = []
        try_to_grow = []
        try_to_complete = []

        evaluated_states = [[(last_actions+'WAIT', state.copy(), self.grade_state(state.copy(), 0))], [], [], []]


        for tree in state.trees[me]:
            # Try seeding
            if not 'SEED' in last_actions_list:
                for cell_id in state.nodes_around(tree, state.cells[tree].tree):
                    if state.can_plant(me, tree, cell_id, cost=seed_cost)[0]:
                        try_state = state.copy()
                        try_state.plant(me, tree, cell_id, cost=seed_cost)
                        evaluated_states[1].append((last_actions+'SEED '+str(tree)+' '+str(cell_id), try_state, self.grade_state(try_state, 1)))
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
    
    def play_turn(self, verbose = False):
        actions_0 = self.player_0.play(self.get_copy_state()).split(' | ')
        actions_1 = self.player_1.play(self.get_copy_state()).split(' | ')
        
        action_0 = actions_0[0]
        action_1 = actions_1[0]

        if verbose:
            print("Player 0", action_0)
            print("Player 1", action_1)

        if action_0[0] == 'W' and action_1[0] == 'W':
            self.state.next_day()
        else:
            self.play_actions([action_0, action_1])
    
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
    
    def play_game(self, verbose=False):
        self.state.next_day()
        while self.state.day < 24:
            if verbose:
                print(" ")
                print('Day',self.state.day, ", Suns : ", self.state.suns, ", Points : ", self.state.points)
                print([(t, self.state.cells[t].tree) for t in self.state.trees[0]])
                print([(t, self.state.cells[t].tree) for t in self.state.trees[1]])
            self.play_turn(verbose)


##########################
# MAIN             #######
##########################


def main():
    player_0 = IA_1(0)
    player_1 = Player(1)
    print(Game(player_0, player_1).play_game(verbose=True))

    
def train_IA_2():

    n_population = 50
    n_fights = 10
    n_generations = 100
    n_repro = 10

    players = [IA_2(666) for i in range(n_population)]

    for generation in range(n_generations):

        scores = [[i, 0] for i in range(n_population)]
        for p_0_idx, p_0 in enumerate(players):
            for i in range(n_fights):
                #print(p_0_idx, i)
                p_1_idx = random.randint(0, n_population-1)
                p_1 = players[p_1_idx]
                if p_1 != p_0:
                    p_0.index = 0
                    p_1.index = 1
                    winner, points_0, points_1 = Game(p_0, p_1).play_game()
                    scores[p_0_idx][1] += points_0 + (15 if winner == 0 else -15)
                    scores[p_1_idx][1] += points_1 + (15 if winner == 1 else -15)
        
        scores.sort(key=lambda tup : tup[1], reverse=True)

        print(scores)
        
        new_players = []

        for i in range(n_repro):
            for parameters in [(0, 0), (10, 0.01), (20, 0.02), (30, 0.015)]:
                new_play = players[scores[i][0]].copy()
                new_play.mutate(parameters[0], parameters[1])
                new_players.append(new_play)
                   
        while len(new_players) < n_population:
            new_players.append(IA_2(666))
        
        players = new_players
    
        for i in range(1):
            np.savetxt('weights_0.csv', players[i].network.weights[0], delimiter=',')
            np.savetxt('weights_1.csv', players[i].network.weights[1], delimiter=',')
            np.savetxt('biases_0.csv', players[i].network.biases[0], delimiter=',')
            np.savetxt('biases_1.csv', players[i].network.biases[1], delimiter=',')



if __name__ == "__main__":
    main()
