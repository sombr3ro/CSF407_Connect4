from gameEnv import gameEnv
from MCTS import MCTS_agent
from Q_learn import Q_learn_agent
import numpy as np

#Parameters #############################################

gzip_file_name = "Q_learn.dat.gz"
board_rows = 4
board_cols = 5
test_epoch = 10
train_epochs = 100

#Q_learn agent parameters
n_min_train = 25
n_max_train = 40
n_min_test = 0
n_max_test = 25
batch_epochs = 10

#########################################################

if __name__=='__main__':
    
    Q_agent = Q_learn_agent(player=2, initial_Q_value=0)
    Q_agent.load_Q_table(gzip_file_name)
    game = gameEnv(height=board_rows, width=board_cols, win_streak=4)
    wins = 0
    losses = 0
    stalemates=0

    for e in range(train_epochs):
        n = np.random.randint(n_min_train, n_max_train+1)
        MCTS_player = MCTS_agent(playouts=n, player=1)
        w,l,s = Q_agent.train(epoch=batch_epochs, agent= MCTS_player, game = game, alpha=0.1, gamma=0.9, greedy_prob=0.05)
        wins+=w
        losses+=l
        stalemates+=s

        if ((e+1)%10 == 0):
            print(f"Epoch {e+1} train performance")
            print(f"Wins: {wins}\t Losses: {losses}\t Stalemates: {stalemates}")
            print(f"Size of Q-table {Q_agent.get_Q_table_size()}")
            wins=0
            losses=0
            stalemates=0

    Q_agent.save_Q_table(gzip_file_name)

    wins=0
    losses=0
    stalemates=0

    for t in range(test_epoch):
        n = np.random.randint(n_min_test, n_max_test+1)
        MCTS_player = MCTS_agent(playouts=n, player=1)
        w,l,s = Q_agent.test(epoch=batch_epochs, agent=MCTS_player, game = game)
        wins+=w
        losses+=l
        stalemates+=s 
    
    print("\nTest Performance")
    print(f"Wins: {wins}\t Losses: {losses}\t Stalemate: {stalemates}")


    
    

