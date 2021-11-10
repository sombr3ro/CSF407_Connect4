import numpy as np

class gameEnv:
    def __init__(self, env_copy = None,  height = None, width = None, win_streak = 4):

        if (env_copy == None):
            self.win_streak = 4
            self.h = height 
            self.w = width
            self.grid = np.zeros((height, width), dtype=int)
        else:
            self.h = env_copy.h
            self.w = env_copy.w
            self.grid = env_copy.grid.copy()
            self.win_streak = env_copy.win_streak
    
    def check_valid_move(self, action):
        if (self.grid[self.h-1][action-1] != 0):
            return False
        else:
            return True

    def get_action_space(self):

        actions = []
        for i in range(1,self.w+1):
            if self.check_valid_move(i):
                actions.append(i)
        return actions
    
    def make_move(self,action, player):

        if not(self.check_valid_move(action)):
            assert("Not a valid move")
        
        insert_pos = -1
        for i in range(self.h):
            if(self.grid[i][action-1]==0):
              self.grid[i][action-1] = player
              insert_pos = i
              break

        if(self.victory_move(action-1, insert_pos, player)):
            return 1
        
        for i in range(self.w):
            if self.grid[self.h-1][i] == 0:
                return 0
        
        return 2

    def victory_move(self, x, y, player):
        #Finds if the current action has resulted in a victory move
    
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
        i = max(0, x - self.win_streak-1)
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
        for i in range(self.h-1,-1,-1):
            for j in range(self.w):
                print(self.grid[i][j], end = " ")
            print()
        pass


if __name__=='__main__':

    game = gameEnv(height = 3,width = 3, win_streak=3)
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

        