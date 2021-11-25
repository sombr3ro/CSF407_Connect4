import numpy as np
from gameEnv import gameEnv
from agent import Agent

#### Parameters ######################################################

height = 6
width = 5
mcts_C = 2
player1_playouts = 40
player2_playouts2 = 200

######################################################################

class Node:
    '''
        Implements Nodes for trees used in Monte Carlo Tree Search
    '''

    def __init__(self):
        self.total_trials = 0   
        self.player = 1     #1: player, -1: Adversary
        self.reward = 0     
        self.children = dict()          #Stores edges as dictionary: key->action, value->node reached by the action
        self.is_leaf = True             #Boolean value to store if node is a leaf
        self.terminal_state = 0         #Terminal States: 0: Not a terminal state, 1: Terminal victory, -1: Stalemate

class MCTS_agent(Agent):
    '''
        Contains implementation of Monte Carlo Tree Search algorithm to play the game Connect4
    '''

    def __init__(self, player, playouts = 20, C = 1):
        super().__init__(player)
        self.playouts = playouts
        self.C = C 
        self.root_node = Node()
        self.first_move = False             #Boolean value that stores if the first action has been performed

    def get_next_action(self, game_env):
        '''
            Performs Monte Carlo Tree Search and returns the next best action
            Arguments:
                game_env -> game environment
            Returns:
                next_move -> best move according to the agent
        '''

        for p in range(self.playouts):
            simulation_env = gameEnv(game_env)
            top_node = self.root_node

            top_node,path = self.selection(top_node, simulation_env)
            child = self.expand_node(top_node, path, simulation_env)

            result = self.simulate(child, simulation_env)
            self.backpropagate(path, result)

        if not self.first_move:                     #If first move has not been performed, perform the first move
            self.first_move = True

        next_move = self.next_best_move(self.root_node, game_env)
    
        return next_move

    def print_tree_details(self):
        '''
            Prints debugging details regarding the tree constructed
        '''
        print(f"Total Size of the tree is: {self.size_of_tree(self.root_node)}")
        print(f"Total depth of the tree is: {self.depth_of_tree(self.root_node)}")
        print(f"Average rewards for each future action")
        for action,child in self.root_node.children.items():
            print(f"{action}: {child.reward}/{child.total_trials}", end = "\t")
        print()
        pass
    
    def UCB1(self, node):
        '''
            Returns the child node with the best UCB1 value
            Arguments:
                node: parent node whose child has to be selected
            Returns:
                next_action: action that leads to child node with the best UCB value
                next_node: child node with the best UCB value 
        '''
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
        '''
            Performs selection for MCTS algorithm using UCB based policy
            Arguments:
                node: node from which the selection process starts
                env: environment over which game is played
            Returns:
                node: leaf node selected by the selection algorithm
                path: Array containing all the nodes in the tree path taken
        '''
        path = []
        path.append(node)
        while (not node.is_leaf ) and (node.terminal_state==0):
            action, child = self.UCB1(node)
            path.append(child)
            child.terminal_state = env.make_move(action, self.get_player_val(node.player))
            node = child

            #To limit the path to depth 4 during the first run
            if (not self.first_move ) and (len(path)==5):           
                break
        
        return node,path

    def expand_node(self,node,path,env):
        '''
            Expands the input leaf node and selects a random child of it to perform the playout on
            Arguments:
                node: leaf node that is expanded
                path: tree path taken by the MCTS algo
                env: Game environment
            Returns:
                node: Random child of the leaf node after expansion 
        '''
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
        '''
            Performs the simulation algorithm by playing out the game on the input node by randomly sampling actions
            from the action space until it reaches a terminal state
            Arguments:
                node: leaf node on which simulation is performed
                env: game environment
            Returns:
                reward: final reward from the game
        '''
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
        '''
            Backpropagates the reward over the tree path taken and updates all the nodes in the path
            Argument:
                path: array containing the nodes visited
                result: final reward of the game
        '''
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

    def next_best_move(self,node,game_env):
        '''
            Returns the next best action according to the MCTS algorithm after a single evaluation
            Argument:
                node: root node from which action has to be chosen
                game_env: Game Environment
            Return:
                best_action: action that leads to the most visited child node (best action according to MCTS)
        '''
        best_action = None
        moves = 0
        
        #If the node has no children, randomly sample an action from the action space
        if not node.children:
            return np.random.choice(game_env.get_action_space())
        
        for action,child in node.children.items():
            if (moves < child.total_trials):
                moves = child.total_trials
                best_action = action
            if (child.terminal_state==1):
                return action
        
        return best_action

    def update_agent_state(self,action):
        '''
            Updates the root node of the Monte Carlo Tree based on the action taken
            Arguments:
                action: Action taken
        '''

        #If the root node does not have any children or has no child node associated with that action
        if not(self.root_node.children and (action in self.root_node.children)):
            self.root_node = Node()
            return

        self.root_node = self.root_node.children[action]
        pass

    def size_of_tree(self,node):
        '''
            Calculates the total size of the tree rooted at the input node
            Argument:
                node: Root node of the tree to be evaluated
            Returns:
                total_size: Total number of nodes in the tree rooted at input node
        '''
        total_size = 1
        for child in node.children.values():
            total_size+= self.size_of_tree(child)
        return total_size 

    def depth_of_tree(self,node):
        '''
            Calculates the depth of tree
            Argument:
                node: Root node from which depth is measured
            Returns:
                depth: depth of the tree rooted at the input node 
        '''
        depth = 1
        for child in node.children.values():
            depth = max(depth,self.depth_of_tree(child)+1)
        return depth 

    def get_player_val(self,player):
        '''
            Gets the value of the player
        '''
        if (player==1):
            return self.player
        else:
            if(self.player==1):
                return 2
            else:
                return 1
    
    def get_action_value(self, game_env, action):
        '''
            Gets the value associated with the input action over the game
            Arguments:
                game_env: Game environment
                action: Action to be evaluated
            Returns:
                value: Value associated with the input action
        '''
        node = self.root_node.children[action]
        value = 0
        
        if (node.total_trials > 0):
            value = node.reward / node.total_trials
        
        return value

    def reset_agent(self):
        '''
            Resets the MCTS agent for a new game
        '''
        self.root_node = Node()
        self.first_move = False

if __name__=='__main__':

    game = gameEnv(height=6,width=5,win_streak=4)
    comp_play_1 = MCTS_agent(playouts=200, player=1, C=1)
    comp_play_2 = MCTS_agent(playouts=40, player=2, C=1)

    human_player = False
    debug = False
    verbose = False
    total_runs = 10

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
                move1 = comp_play_1.get_next_action(game)
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
                comp_play_1.update_agent_state(abs(move1))
            comp_play_2.update_agent_state(abs(move1))

            if (human_player or debug or verbose):
                print(f"Player 2, make a move")
            
            if(move1>0):
                move2 = comp_play_2.get_next_action(game)
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
            comp_play_1.update_agent_state(abs(move2))
            comp_play_2.update_agent_state(abs(move2))
        
        game.reset_game()
        comp_play_1.reset_agent()
        comp_play_2.reset_agent()
        
    print(f"Player 1 wins: {player1_wins}")
    print(f"Player 2 wins: {player2_wins}")
    print(f"Stalemates: {stalemates}")
    