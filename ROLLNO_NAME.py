
##############

#Your program can go here.

###############

def PrintGrid(positions):
    print('\n'.join(' '.join(str(x) for x in row) for row in positions))
    print()

def main():
    
    
    print("************ Sample output of your program *******")

    game1 = [[0,0,0,0,0],
          [0,0,0,0,0],
          [0,0,1,0,0],
          [0,2,2,0,0],
          [1,1,2,2,0],
          [2,1,1,1,2],
        ]


    game2 = [[0,0,0,0,0],
          [0,0,0,0,0],
          [0,0,1,0,0],
          [1,2,2,0,0],
          [1,1,2,2,0],
          [2,1,1,1,2],
        ]

    
    game3 = [ [0,0,0,0,0],
              [0,0,0,0,0],
              [0,2,1,0,0],
              [1,2,2,0,0],
              [1,1,2,2,0],
              [2,1,1,1,2],
            ]

    print('Player 2 (Q-learning)')
    print('Action selected : 2')
    print('Value of next state according to Q-learning : .7312')
    PrintGrid(game1)


    print('Player 1 (MCTS with 25 playouts')
    print('Action selected : 1')
    print('Total playouts for next state: 5')
    print('Value of next state according to MCTS : .1231')
    PrintGrid(game2)

    print('Player 2 (Q-learning)')
    print('Action selected : 2')
    print('Value of next state : 1')
    PrintGrid(game3)
    
    print('Player 2 has WON. Total moves = 14.')
    
if __name__=='__main__':
    main()