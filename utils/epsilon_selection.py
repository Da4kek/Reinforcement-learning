import numpy as np 
import gym 

def epselection(epsilon,q_table,discrete_state,env):
    random_number = np.random.random()
    if random_number > epsilon:
        state_row = q_table[discrete_state,:]
        action = np.argmax(state_row)
    else:
        action = env.action_space.sample()
    return action