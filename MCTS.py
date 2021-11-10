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
    1) Implement the limitation of depth 4 for the first move
    2) Create a function that returns the average reward of the action chosen
    3) For debugging purposes, try to implement a way to visualize the graph
    4) Improve Documentation
'''

class Node:

    def __init__(self):
        self.total_trials = 0
        self.player = 1     #1: player, -1: Adversary
        self.reward = 0
        self.children = dict()
        self.is_leaf = True
        self.terminal_state = 0         #Terminal States: 0: Not a terminal state, 1: Terminal victory, -1: Stalemate

class MCTS:

    def __init__(self, playouts = 20, C = 1):
        self.playouts = playouts
        self.C = C 
        self.root_node = Node()

    def single_run(self, game_env):
        #Performs a single evaluation of the game and return the next best move

        for p in range(self.playouts):
            simulation_env = gameEnv(game_env)
            top_node = self.root_node
            
            top_node,path = self.selection(top_node, simulation_env)
            child = self.expand_node(top_node, path, simulation_env)

            result = self.simulate(child, simulation_env)
            self.backpropagate(path, result)
        
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
        #Performs selection using UCB based policy
        path = []
        path.append(node)
        while not node.is_leaf and node.terminal_state==0:
            action, child = self.UCB1(node)
            path.append(child)
            child.terminal_state = env.make_move(action, node.player)
            node = child
        
        return node,path

    def expand_node(self,node,path,env):
        #Expands a node and selects a random child of it to playout on
        if not(node.terminal_state==0):
            return node

        action_space = env.get_action_space()
        for a in action_space:
            node.children[a] = Node()
            node.children[a].player = -1*node.player

        node.is_leaf = False
        next_action = np.random.choice(action_space)
        node = node.children[next_action]
        path.append(node)
        node.terminal_state = env.make_move(next_action, -1*node.player)
        return node

    def simulate(self,node,env):
        #Performs the simulation function
        if not(node.terminal_state==0):
            return node.player*node.terminal_state

        player = node.player
        reward = node.terminal_state
        
        while reward==0:
            next_action = np.random.choice(env.get_action_space())
            reward = env.make_move(next_action, player)
            player *=-1

        return player*reward

    def backpropagate(self, path, result):
        #Performs backprop after a single playout
        for node in reversed(path):
            if node.player==1 and result==1:
                node.reward+=1
            elif node.player==-1 and result==-1:
                node.reward+=1
            node.total_trials+=1
        pass

    def next_best_move(self,node):
        #Returns the next best action estimated by the MCTS
        best_action = None
        moves = 0
        for action,child in node.children.items():
            if (moves < child.total_trials):
                moves = child.total_trials
                best_action = action
        return best_action

    def update_node(self,action):
        #Updates the root node of the MCTS object based on the action taken
        if not(self.root_node.children):
            return

        self.root_node = self.root_node.children[action]
        pass


if __name__=='__main__':

    game = gameEnv(height=6,width=5,win_streak=4)
    comp_play = MCTS()

    game.print_grid()

    while True:

        print("Human player, make a move")
        move = int(input())
        res = game.make_move(move,-1)
        game.print_grid()

        if not(res==0):
            if(res==2):
                print("Stalemate")
            else:
                print("Hooman wins ;(")
            break

        comp_play.update_node(move)

        comp_move = comp_play.single_run(game)
        print(f"My chance, HumAn, lemme try {comp_move}")
        res = game.make_move(comp_move,1)
        game.print_grid()

        if not(res==0):
            if(res==2):
                print("Stalemate")
            else:
                print("Me win ;)")
            break
        comp_play.update_node(move)
    
        
        

