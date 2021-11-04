import numpy as np
from gameEnv import gameEnv

# Parameters --------------------------------------------------------

height = 6
width = 5
mcts_C = 2
player1_playouts = 40
player2_playouts2 = 200

# -------------------------------------------------------------------

'''
    TO DO
    1) Implement case of terminal states in each function
    2) Complete all functions
'''

class Node:

    def __init__(self):
        self.total_trials = 0
        self.reward = 0
        self.children = dict()
        self.is_leaf = True

class MCTS:

    def __init__(self, playouts = 20, C = 1):
        self.playouts = playouts
        self.C = C 
        self.root_node = Node()

    def single_run(self, game_env):

        for p in range(self.playouts):
            simulation_env = gameEnv(game_env)
            top_node = self.root_node
            
            top_node,path = self.select(top_node, simulation_env)
            child = self.expand_node(top_node, path, simulation_env)

            result = self.simulate(child, simulation_env)
            self.backpropogate(path, result)
        
        return self.next_best_move(self.root_node)

    
    def UCB1(self, node):
        #Returns the node with the best UCB value
        max_val = - np.inf
        next_node = None
        next_action = None

        for action,child in node.children.items():

            if child.total_trials == 0:
                return action,child
            
            val = child.reward/child.total_trials + self.C* np.sqrt( np.log(node.total_trials)/ child.total_trials)
            if(val > max_val):
                max_val = val
                next_node = child
                next_action = action
        
        return next_action, next_node

    def selection(self,node, env):
        
        path = []
        path.append(node)
        while not node.is_leaf:
            action, child = self.UCB1(node)
            path.append(child)
            env.make_move(action)
            node = child
        
        return node,path

    def expand_node(self,node,path,env):

        action_space = env.get_action_space()
        for a in action_space:
            node.children[a] = Node()

        next_action = np.random.choice(action_space)
        node = node.children[next_action]
        path.append(node)
        env.make_move(next_action)
        return node

    def simulate(self,node,env):
        pass

    def backpropagate(self, path, result):
        pass

    def next_best_move(self,node):
        pass
    
