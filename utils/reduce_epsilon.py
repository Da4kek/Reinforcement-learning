import numpy as np 
def reduce_epsilon(epsilon,epoch,min_epsilon,max_epsilon,decay_rate):
    return min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * epoch)