import time 
import numpy as np
from utils.epsilon_selection import epselection
from utils.reduce_epsilon import reduce_epsilon
from utils.q_selection import computeQ
from utils.show_state import show_state

class Discrete():
    def __init__(self,env):
        self.env = env
        self.action_size = self.env.action_space.n
        self.state_size = self.env.observation_space.n
    
    def q_table(self,action_size,state_size):
        qtable = np.zeros([state_size,action_size])
        return qtable
    
    def fit(self,epochs,learning_rate,gamma,epsilon,decay_rate,min_epsilon,max_epsilon):
        q_table = self.q_table(self.action_size,self.state_size)
        for epoch in range(epochs):
            discrete_state = self.env.reset()
            done = False
            while not done:
                action = epselection(epsilon,q_table,discrete_state,self.env)
                new_state,reward,done,info = self.env.step(action)
                old_q_value = q_table[discrete_state,action]
                next_optimal_q = np.max(q_table[new_state,:])

                next_q = computeQ(old_q_value,reward,next_optimal_q,learning_rate,gamma)

                q_table[discrete_state,action] = next_q
                discrete_state = new_state
            epoch +=1
            epsilon = reduce_epsilon(epsilon,epoch,min_epsilon,max_epsilon,decay_rate)
        return q_table

    def evaluate(self,size,visualize:bool=False):
        state = self.env.reset()
        for _ in range(size): 
            self.env.render()
            action = np.argmax(self.q_table(self.action_size,self.state_size)[state])
            state,reward,done,info = self.env.step(action)
            time.sleep(1)
            if visualize:
                show_state(self.env)
            if done:
                break 
        self.env.close()
        return reward,info
    