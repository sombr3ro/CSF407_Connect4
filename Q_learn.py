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
    
    def set_Qvalue(self,state, action, player, value):
        '''
            Refers to the after-state implementation and stores it in that form
        '''

        temp_game = gameEnv(env_copy=state)
        temp_game.make_move(action,player)
        key = temp_game.generate_string()

        self.Q_table[key] = value
        pass
    
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
    
    def get_reward(self,terminal_state):
        #Returns reward according to the terminal state
        terminal_reward = 50
        if(terminal_state==0):
            reward = -1
        elif(terminal_state==1):
            reward = terminal_reward
        else:
            return 0
        return reward

    def train(self, epoch=100, agent = None, game = None):
        #To be completed


        for e in epoch:
            while True:
                move1 = agent.single_run(game)
                state  = game.make_move(move1,1)
                reward = self.get_reward(state)
                agent.update_node(move1)

                #Action selection using e-greedy policy
                action_space = game.get_action_space()
                if (np.random.random() < self.greedy_prob):
                    move2 = np.random.choice(action_space)
                else:
                    best_action = None
                    best_q_val = - np.inf
                    for a in action_space:
                        q_val = self.get_Qvalue(game,a,2)
                        if(q_val > best_q_val):
                            best_q_val = q_val
                            best_action = a
                    move2 = best_action
                
                #Perform move 2
                state = game.make_move(move2,2)
                agent.update_node(move2)
                reward

                #Q(S,a) update step
                key = game.generate_string()
                action_space = game.get_action_space()

        pass

if __name__=='__main__':
    agent = Q_learning_algo()
    agent.Q_table["12312321321321"] = 11
    agent.Q_table["3216567321763"] = 202
    print(agent.Q_table)
    agent.save_Q_table("random.dat")
    agent2 = Q_learning_algo()
    agent2.load_Q_table("random.dat")
    print(agent2.Q_table)
    


    

        

