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

    def __init__(self, playouts = 20, C = 1, player=2):
        self.playouts = playouts
        self.C = C 
        self.root_node = Node()
        self.player = player
        self.first_move = False             #Boolean value that stores if the first action has been performed

    def single_run(self, game_env):
        #Performs a single evaluation of the game and return the next best move
        
        #self.root_node.terminal_state = False       #No idea why

        for p in range(self.playouts):
            simulation_env = gameEnv(game_env)
            top_node = self.root_node

            top_node,path = self.selection(top_node, simulation_env)
            child = self.expand_node(top_node, path, simulation_env)

            result = self.simulate(child, simulation_env)
            self.backpropagate(path, result)

        if not self.first_move:                     #If first move has not been performed, perform the first move
            self.first_move = True

        #print(game_env.get_action_space())
        #self.print_tree_details()
        
        next_move = self.next_best_move(self.root_node)

        '''
        if(next_move==None):
            #print(s)
            print(f"terminal state {self.root_node.terminal_state}")
            print(self.root_node.is_leaf)
            print(self.root_node.total_trials)
            print(self.root_node.reward)
            self.print_tree_details()
            game_env.print_history()
            game_env.print_grid()
        '''

        return next_move

    def print_tree_details(self):
        print(f"Total Size of the tree is: {self.size_of_tree(self.root_node)}")
        print(f"Total depth of the tree is: {self.depth_of_tree(self.root_node)}")
        print(f"Average rewards for each future action")
        for action,child in self.root_node.children.items():
            print(f"{action}: {child.reward}/{child.total_trials}", end = "\t")
        print()
        pass
    
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
        while (not node.is_leaf ) and (node.terminal_state==0):
            action, child = self.UCB1(node)
            path.append(child)
            child.terminal_state = env.make_move(action, self.get_player_val(node.player))
            node = child

            if (not self.first_move ) and (len(path)==5):           #To limit the path to depth 4 during the first run
                break
        
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
        node.terminal_state = env.make_move(next_action, self.get_player_val(-1*node.player))
        return node

    def simulate(self,node,env):
        #Performs the simulation function
        if not(node.terminal_state==0):
            return node.player*node.terminal_state

        player = node.player
        reward = node.terminal_state
        
        while reward==0:
            next_action = np.random.choice(env.get_action_space())
            reward = env.make_move(next_action, self.get_player_val(player))
            player *=-1

        return player*reward

    def backpropagate(self, path, result):
        #Performs backprop after a single playout
        for node in reversed(path):
            if result==1:
                if node.player==1:
                    node.reward+=1
                else:
                    node.reward-=1
            elif result==-1:
                if node.player==1:
                    node.reward-=1
                else:
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
            if (child.terminal_state==1):
                return action
        return best_action

    def update_node(self,action):
        #Updates the root node of the MCTS object based on the action taken
        if not(self.root_node.children and (action in self.root_node.children)):
            self.root_node = Node()
            return

        self.root_node = self.root_node.children[action]
        pass

    def size_of_tree(self,node):
        #Calculates total size of the tree
        total_size = 1
        for child in node.children.values():
            total_size+= self.size_of_tree(child)
        return total_size 

    def depth_of_tree(self,node):
        depth = 1
        for child in node.children.values():
            depth = max(depth,self.depth_of_tree(child)+1)
        return depth 

    def get_player_val(self,player):
        #Returns the player id to return to the game world object

        if (player==1):
            return self.player
        else:
            if(self.player==1):
                return 2
            else:
                return 1

    def reset_agents(self):
        #Reset the MCTS trees for a new game
        self.root_node = Node()
        self.first_move = False

if __name__=='__main__':

    game = gameEnv(height=6,width=5,win_streak=4)
    comp_play_1 = MCTS(playouts=200, player=1, C=1)
    comp_play_2 = MCTS(playouts=40, player=2, C=1)

    human_player = True
    debug = False
    verbose = True
    total_runs = 1

    player1_wins = 0
    player2_wins = 0
    stalemates = 0
    for i in range(total_runs):
        while True:

            if (human_player):
                print("Player 1, make a move")
                move1 = int(input())
            else:
                if (verbose):
                    print("Player 1 makes a move")
                move1 = comp_play_1.single_run(game)
            res = game.make_move(move1,1, track_history=debug)
            
            if (human_player or debug or verbose):
                print(f"Lemme try {move1}")
                game.print_grid()

            if not(res==0):
                if(res==2):
                    if (verbose):
                        print("Stalemate")
                    stalemates+=1
                else:
                    if (verbose):
                        print("Player 1 wins ;(")
                    player1_wins+=1
                break

            if not human_player:
                comp_play_1.update_node(abs(move1))
            comp_play_2.update_node(abs(move1))

            if (human_player or debug or verbose):
                print(f"Player 2, make a move")
            
            if(move1>0):
                move2 = comp_play_2.single_run(game)
            else:
                move2 = int(input())

            res = game.make_move(move2,2, track_history=debug)
        
            if (human_player or debug or verbose):
                print(f"Lemme try {move2}")
                print()
                game.print_grid()

            if not(res==0):
                if(res==2):
                    if (verbose):
                        print("Stalemate")
                    stalemates+=1
                else:
                    if (verbose):
                        print("Player 2 win ;)")
                    player2_wins+=1
                break
            comp_play_1.update_node(abs(move2))
            comp_play_2.update_node(abs(move2))
        
        game.reset_game()
        comp_play_1.reset_agents()
        comp_play_2.reset_agents()
        
    print(f"Player 1 wins: {player1_wins}")
    print(f"Player 2 wins: {player2_wins}")
    print(f"Stalemates: {stalemates}")
    