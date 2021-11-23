from os import stat
import numpy as np
from gameEnv import gameEnv
import gzip
import json
from MCTS import MCTS

class Q_learn_agent:

    def __init__(self, player=2):
        self.player = player
        self.Q_table = {}
        self.initial_Q_value = 0

    def get_Qvalue(self, state, action, player):
        '''
        We will use the concept of after-states to calculate the Q(S,a) value
        First the board will perform the action a, then the state value function of the resultant state
        will be looked up
        '''

        #Perform the action
        temp_game = gameEnv(env_copy=state)
        temp_game.make_move(action, player)
        return self.get_Qvalue_afterstate(temp_game)
    
    def get_Qvalue_afterstate(self,state):
        #Get Q_Value from after state as an argument
        key = state.generate_string()
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
        with gzip.open(file_name,'w') as fout:
            fout.write(json.dumps(self.Q_table).encode('utf-8'))

    def load_Q_table(self, filename):
        #Load Q_table from the file
        with gzip.open(filename,'r') as fin:
            self.Q_table = json.loads(fin.read().decode('utf-8'))
    
    def get_reward(self,terminal_state, player):
        #Returns reward according to the terminal state
        terminal_reward = 100
        if(terminal_state==0):
            reward = 0
        elif(terminal_state==1):
            reward = terminal_reward
            if (player== -1):           #Opponent Player condtion
                reward *= -1
        else:
            return 10
        return reward

    def train(self, epoch=100, agent = None, game = None, alpha = 0.1, gamma = 0.99, greedy_prob=0.1):

        wins = 0
        stalemates = 0 
        losses=0
        total_plays = 0
        for e in range(epoch):
            
            total_plays +=1

            #External Agent makes the first move
            move1 = agent.single_run(game)
            game.make_move(move1,1)
            agent.update_node(move1)

            while True:

                state1 = -1
                state2 = -1
                
                #Action selection using e-greedy policy
                action_space = game.get_action_space()
                if (np.random.random() < greedy_prob):
                    move1 = np.random.choice(action_space)
                else:
                    best_action = None
                    best_q_val = - np.inf
                    for a in action_space:
                        q_val = self.get_Qvalue(game,a,2)
                        if(q_val > best_q_val):
                            best_q_val = q_val
                            best_action = a
                    move1 = best_action
                
                #Q_learn agent performing move 1
                state1 = game.make_move(move1,2)
                agent.update_node(move1)            #Agent updating it's state
                move1_reward = self.get_reward(state1,1)
                key = game.generate_string()
                old_q_value = self.get_Qvalue_afterstate(game)

                #External agent performing move2
                if (state1==0):
                    move2 = agent.single_run(game)
                    state2  = game.make_move(move2,1)
                    move2_reward = self.get_reward(state2,-1)
                    agent.update_node(move2)
                else:
                    move2_reward = 0
                    if (state1==1):
                        wins+=1
                        #print("Win by player2")
                    elif not(state1 == -1):
                        stalemates +=1
                        #print("Stalemate first half")

                #Q(S,a) update step

                total_reward = move1_reward + move2_reward

                best_q_val = - np.inf

                if (state2==0):
                    action_space = game.get_action_space()
                    for a in action_space:
                        q_val = self.get_Qvalue(game,a,2)
                        if( q_val > best_q_val):
                            best_q_val = q_val
                else:
                    best_q_val = 0
                    if(state2==1):
                        losses+=1
                        #print("Win by player1")
                    elif not state2==-1:
                        stalemates+=1
                        #print("Stalemate second half")
                
                self.Q_table[key] = old_q_value + alpha*(total_reward + gamma*best_q_val - old_q_value)

                if (not (state1 == 0)) or (not (state2 ==0)):
                    break

            
            if((e+1)%100 == 0):
                print(f"Train epoch {e+1}: Wins: {wins}/{total_plays}\t losses: {losses}/{total_plays}\t Stalemates: {stalemates}/{total_plays}")
                print(f"Q_table size: {len(self.Q_table)}")
                total_plays=0
                wins=0
                stalemates=0
                losses=0
            
            
            game.reset_game()
            agent.reset_agents()

        self.save_Q_table(f"Connect{game.h}x{game.w}learner_a{alpha}_g{gamma}.dat.gz")
        #print(f"Alpha {alpha}, Gamma {gamma} \t Wins: {wins}/{total_plays}\t losses: {losses}/{total_plays}\t Stalemates: {stalemates}/{total_plays}")
        pass

    def test(self, epoch=100, agent = None, game = None):
        wins = 0
        losses = 0
        total_plays = 0
        stalemates = 0
        

        for e in range(epoch):

            game.reset_game()
            agent.reset_agents()

            player_turn = 1
            total_plays+=1

            while True:
                
                if (player_turn==1):
                    move = agent.single_run(game)
                else:
                    move = self.single_run(game)
                
                state = game.make_move(move,player_turn)
                agent.update_node(move)

                if not(state == 0):
                    if (state == 1):
                        if (player_turn==1):
                            losses+=1
                        else:
                            wins+=1
                    else:
                        stalemates+=1
                    break

                player_turn = 2 - (player_turn+1)%2
            
        
        print(f"Test run: Wins: {wins}/{total_plays}\t losses: {losses}/{total_plays}\t Stalemates: {stalemates}/{total_plays}")


    def single_run(self, game):
        #Return the best move according to the Q-values 

        action_space = game.get_action_space()
        best_q_val = -np.inf
        best_action = None
        for a in action_space:
            q_val = self.get_Qvalue(game,a,self.player)
            if (q_val > best_q_val):
                best_q_val = q_val
                best_action = a
        
        return best_action


if __name__=='__main__':

    '''
    agent = Q_learning_algo()
    agent.Q_table["12312321321321"] = 11
    agent.Q_table["3216567321763"] = 202
    print(agent.Q_table)
    agent.save_Q_table("random.dat")
    agent2 = Q_learning_algo()
    agent2.load_Q_table("random.dat")
    print(agent2.Q_table)
    '''

    MCTS_agent = MCTS(playouts=40,player=1)
    Q_learner = Q_learn_agent(player=2)
    game = gameEnv(height=5,width=4, win_streak=4)
    #Q_learner.load_Q_table("Connect6x5learner_a0.1_g0.95.dat.gz")
    Q_learner.train(epoch=2000,agent=MCTS_agent, game = game, alpha=0.1, gamma=0.95, greedy_prob=0.05) 
    #Q_learner.test(epoch=100, agent=MCTS_agent, game= game)

    '''
    for alpha in np.arange(0.5,10,0.5):
        for gamma in np.arange(0.1,1.0,0.1):
            Q_learner.train(epoch=1000,agent=MCTS_agent, game = game, alpha=alpha, gamma=gamma)
    '''


    


    

        

