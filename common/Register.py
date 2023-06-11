import gym 
import time

class Register():
    def __init__(self):
        self.env = None
        self.env_name = None
    
    def  Make_env(self,env_name):
        try:
            self.env = gym.make(env_name)
            self.env_name = env_name
            return self.env
        except KeyError:
            raise KeyError("Please register with a valid environment name")
    

    def show_states(self):
       
        self.env.reset()
        for _ in range(10):
            self.env.render(mode="human")
            observation,reward,done,info = self.env.step(self.env.action_space.sample())
            time.sleep(0.2)
            if done:
                break
        return observation,reward,info
        
        