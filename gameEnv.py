import numpy as np
import sys

class gameEnv:
    '''
        Implements an environment class over which the Connect 4 game can be played
    '''
    def __init__(self, env_copy = None,  height = None, width = None, win_streak = 4):
        '''
            Initializes the gameEnv object
            Arguments:
                env_copy: If not None, the object copies the 'env_copy' gameEnv object
                height: Number of rows in the Connect 4 board
                width: Number of columns in the Connect 4 board
                win_streak: Continuous number of beads that is considered a victory in the game 
        '''
        if (env_copy == None):  
            #Copies the env_copy gameEnv object
            self.win_streak = win_streak
            self.h = height 
            self.w = width
            self.grid = np.zeros((height, width), dtype=int)
        else:
            self.h = env_copy.h
            self.w = env_copy.w
            self.grid = env_copy.grid.copy()
            self.win_streak = env_copy.win_streak
        
        self.history = []                   #Used to store the history of the game for debugging purposes
        self.action_history = []            #Used to store the action history of the game for debugging purposes
    
    def check_valid_move(self, action, debug = False):
        '''
            Function to check the validity of a move
            Arguments: 
                action: Move whose validity has to be verified
                debug: if set, runs the function in debug mode
            Returns:
                valid: boolean value representing if the action is valid
        '''
        if (action > self.w):
            if (debug):
                print("Illegal action command")
            return False
        
        if (self.grid[self.h-1][action-1] != 0):
            if (debug):
                print(f"Column already filled: {self.h-1}")
                self.print_grid()
            return False
        
        return True

    def get_action_space(self):
        '''
            Returns the current action space of the game environment
            Returns:
                actions: array containing the valid actions
        '''
        actions = []
        for i in range(1,self.w+1):
            if self.check_valid_move(i):
                actions.append(i)
        return actions
    
    def make_move(self,action, player, track_history = False):
        '''
            Performs the action by the specified player on the game
            Arguments:
                action: Action to be performed on the board
                player: player id of the player performing the move
                track_history: if True, stores the history of actions and state grids (for debugging)
            Returns:
                status: Status of the board:
                            0: Transient State
                            1: Victory
                            2: Stalemate
        '''

        TRANSIENT_STATE_STATUS = 0
        VICTORY_STATUS = 1
        STALEMATE_STATUS = 2

        #Check if the action is valid
        if not(self.check_valid_move(action, debug=True)):
            sys.exit(f"Move {action} is not a valid move by player {player}")
        
        insert_pos = -1

        #Update state grid
        for i in range(self.h):
            if(self.grid[i][action-1]==0):
              self.grid[i][action-1] = player
              insert_pos = i
              break
        
        if (track_history):
            self.history.append(self.grid.copy())
            self.action_history.append(action)

        #Check if the board is in a terminal victory state
        if(self.victory_move(action-1, insert_pos, player)):
            return VICTORY_STATUS

        #Check if the board is in a transient state    
        for i in range(self.w):
            if self.grid[self.h-1][i] == 0:
                return TRANSIENT_STATE_STATUS
        
        return STALEMATE_STATUS

    def victory_move(self, x, y, player):
        '''
            Function that checks if the current action has resulted in a victory move
            Arguments:
                x: x value of the last bead dropped
                y: y value of the last bead dropped 
                player: player id of the player
            Returns:
                True: Victory state
                False: Not a victory state
        '''
    
        #vertical check
        j = max(0, y - self.win_streak-1)
        count = 0
        while(j< min(y+self.win_streak, self.h)):
            if(self.grid[j][x]==player):
                count+=1
            else:
                count= 0
            
            if (count==self.win_streak):
                return True
            j+=1
        
        #horizontal check
        i = max(0, x - (self.win_streak-1))
        count=0
        while(i < min(x + self.win_streak, self.w)):
            if(self.grid[y][i]==player):
                count+=1
            else:
                count=0
            
            if(count==self.win_streak):
                return True
            i+=1
        
        
        # positive slope diagonal check
        k = - min( min(x,y), self.win_streak-1)
        k_max = min(min(self.h-y, self.w-x), self.win_streak)
        count = 0
        while(k < k_max):
            if(self.grid[y+k][x+k]==player):
                count+=1
            else:
                count =0
            
            if(count==self.win_streak):
                return True 
            k+=1
        
        # negative slope diagonal check
        k = - min(self.win_streak-1, min(x, self.h-1 - y))
        k_max = min(self.win_streak-1, min(self.w-1-x, y))
        count=0
        while(k <= k_max):
            if(self.grid[y-k][x+k]==player):
                count+=1
            else:
                count =0
            
            if(count==self.win_streak):
                return True 
            k+=1
    
        return False

    def print_grid(self):
        '''
            Prints the game grid
        '''
        for i in range(self.h-1,-1,-1):
            for j in range(self.w):
                print(self.grid[i][j], end = " ")
            print()
        pass

    def generate_string(self):
        '''
            Serialize the grid into a single string
            Returns: 
                s: String that encodes the grid in row-major format
        '''

        s = ""
        for i in self.grid:
            for j in i:
                s = s + str(i)
        return s

    def print_history(self):
        '''
            Function that prints the history of the game grid over the course of the game
        '''
        for curr_grid in self.history:
            for i in range(self.h-1,-1,-1):
                for j in range(self.w):
                    print(curr_grid[i][j], end = " ")
                print()
            print("\n\n")
        print(self.action_history)
        pass

    def reset_game(self):
        '''
            Function to reset the environment 
        '''
        self.grid = np.zeros((self.h, self.w), dtype=int)
        self.history = []
        self.action_history = []


if __name__=='__main__':

    game = gameEnv(height = 4,width = 3, win_streak=3)
    game.print_grid()

    player = 0
    while True:
        
        print(f"Player {player+1}, choose a move:")
        move = int(input())
        
        state = game.make_move(move, player+1)
        game.print_grid()
        
        if (state==1):
            print(f"Player {player+1} has won the game")
            break
        elif (state==-1):
            print("Stalemate")
            break
        
        player = (player+1)%2