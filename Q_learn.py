import numpy as np
from gameEnv import gameEnv
import pickle

class Q_learning_algo:

    def __init__(self, gamma = 0.9, greedy_prob=0.1, player=1):
        self.gamma = gamma
        self.greedy_prob = greedy_prob
        self.player = player
        self.Q_table = {}
        self.initial_Q_value = 1

    def get_Qvalue(self, state, action, player):
        '''
        We will use the concept of after-states to calculate the Q(S,a) value
        First the board will perform the action a, then the state value function of the resultant state
        will be looked up
        '''

        #Perform the action
        temp_game = gameEnv(env_copy=state)
        temp_game.make_move(action, player)
        key = temp_game.generate_string()

        if (key in self.Q_table):
            return self.Q_table[key]
        else:
            self.Q_table[key] = self.initial_Q_value
            return self.initial_Q_value
    
    def save_Q_table(self, file_name):
        #Save the Q_table
        try:
            filehandler = open(file_name, 'wb')
            pickle.dump(self.Q_table, filehandler)
            filehandler.close()
        except:
            print("Failed to save data in the file " + file_name)

    def load_Q_table(self, filename):
        #Load Q_table from the file
        try:
            filehandler = open(filename, 'rb')
            self.Q_table = pickle.load(filehandler)
            filehandler.close()
        except:
            print("Failed to load the file "+filename)

if __name__=='__main__':
    agent = Q_learning_algo()
    agent.Q_table["12312321321321"] = 11
    agent.Q_table["3216567321763"] = 202
    print(agent.Q_table)
    agent.save_Q_table("random.dat")
    agent2 = Q_learning_algo()
    agent2.load_Q_table("random.dat")
    print(agent2.Q_table)
    


    

        

