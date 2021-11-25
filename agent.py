class Agent:
    '''
        Base class for all agents
    '''

    def __init__(self, player):
        self.player = player
        pass

    def get_next_action(self, game_env):
        pass

    def update_agent_state(self, action):
        '''
        Observes the environment and updates the agent
        Arguments:
            action -> Action taken to change the state
        '''
        pass

    def reset_agent(self):
        '''
            Reset the state of the agent
        '''
        pass

    def get_action_value(self, game_env, action):
        '''
            Get the value of the next state achieved by the input action
            Arguments:
                action -> Action whose value has to be estimated
        '''
        pass

