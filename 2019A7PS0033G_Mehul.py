from os import terminal_size
from gameEnv import gameEnv
from Q_learn import Q_learn_agent
from MCTS import MCTS_agent

### Parameters ###################################

game_rows = 4
game_cols = 5
win_streak = 4

# Q_learning agent
qlearn_filename = "Q_learn.dat.gz"

##################################################

if __name__=='__main__':

    print("\nConnect4 game\n")
    print("1) Press 1 for MC_200 vs MC_40")
    print("2_ Press 2 for MC_n vs Q_learn\n")

    choice = int(input("Choose one option: "))

    player1 = None
    player2 = None
    player1_name = ""
    player2_name = ""

    if (choice==1):
        player1 = MCTS_agent(player=1, playouts=200, C=1)
        player1_name = "MCTS agent with 200 playouts"
        player2 = MCTS_agent(player=2, playouts=40, C=1)
        player2_name = "MCTS agent with 40 playouts"
    elif (choice==2):
        player1 = MCTS_agent(player=1, playouts=25, C=1)
        player1_name = "MCTS agent with n playouts"
        player2 = Q_learn_agent(player=2)
        player2.load_Q_table(qlearn_filename)
        player2_name = "Q-learning agent"
    
    game = gameEnv(height=game_rows, width = game_cols, win_streak= win_streak)

    print("\nGame starts\n")
    game.print_grid()
    print()

    turn = 1
    game_status = 0
    moves = 0

    while(game_status==0):

        if(turn == 1):
            action = player1.get_next_action(game)
            action_value = player1.get_action_value(game,action)
            game_status = game.make_move(action, player=1)
            print(f"Player 1: {player1_name}")
            turn = 2
        else:
            action = player2.get_next_action(game)
            action_value = player2.get_action_value(game,action)
            game_status = game.make_move(action, player=2)
            print(f"Player 2: {player2_name}")
            turn = 1

        moves+=1
        print(f"Action selected {action}")
        print(f"Value of action selected according to agent: {action_value}")
        game.print_grid()
        print()
        player1.update_agent_state(action)
        player2.update_agent_state(action)

    if (game_status==1):
        if (turn == 2):
            print("Player 1 has WON")
        else:
            print("Player 2 has WON")
    else:
        print("Stalemate")
    
    print(f"Total moves played = {moves}")
    




    




