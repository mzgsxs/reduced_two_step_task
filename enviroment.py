import random

class reduced_two_step_env():
    def __init__(self, reward_value=1., transition_prob=[1,0], reward_prob=[1,0]):
        self.reward_value = reward_value
        self.transition_prob = transition_prob # probability of transite to good/state from 0 for action 0 and 1
        self.reward_prob = reward_prob # probability of receive/transite to reward 1 form state 1 and 2
        self.reset()
    
    def step(self, action):
        if self.state is 0:
            reward = 0.
            if action is 2:
                new_state = 0
            else:
                if random.random() < self.transition_prob[action]:
                    new_state = 1
                else:
                    new_state = 2
        else:
            if random.random() < self.reward_prob[self.state-1]:
                reward = self.reward_value
            else:
                reward = 0.
            self.end_of_trail = True
            new_state = -1
        self.state = new_state
        return new_state, reward
            
    def reset(self):
        self.state = 0
        self.end_of_trail = False

     
