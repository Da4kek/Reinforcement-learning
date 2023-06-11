import time 
import numpy as np
from utils.epsilon_selection import epselection
from utils.reduce_epsilon import reduce_epsilon
from utils.q_selection import computeQ
from utils.show_state import show_state

class Discrete():
    def __init__(self,env):
        self.action_size = env.action_space.n
        self.state_size = env.observation_space.shape
        self.env = env
    
    def q_table(self):
        qtable = np.zeros([self.state_size,self.action_size])
        return qtable
    
    def fit(self,epochs,learning_rate,gamma,epsilon,decay_rate,min_epsilon,max_epsilon):
        q_table = self.q_table()
        for epoch in range(epochs):
            discrete_state = self.env.reset()
            done = False
            while not done:
                action = epselection(epsilon,q_table,discrete_state,self.env)
                new_state,reward,done,info = self.env.step(action)
                new_discrete_state = new_state
                old_q_value = q_table[discrete_state,action]
                next_optimal_q = np.max(q_table[new_discrete_state,:])
                q_table[discrete_state,action] = computeQ(old_q_value,reward,next_optimal_q,learning_rate,gamma)
                discrete_state = new_discrete_state
            epsilon = reduce_epsilon(epsilon,epoch,min_epsilon,max_epsilon,decay_rate)
        return q_table

    def evaluate(self,size,visualize:bool=False):
        state = self.env.reset()
        for _ in range(size):
            self.env.render()
            action = np.argmax(self.q_table()[state])
            state,reward,done,info = self.env.step(action)
            time.sleep(1)
            if visualize:
                show_state(self.env)
            if done:
                break 
        self.env.close()
        return reward,info
    