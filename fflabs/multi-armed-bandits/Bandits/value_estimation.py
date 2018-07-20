import numpy as np

class Agent_SampleAverage(object): 
    '''
    A collection of functions to implement a Sample Average value estimation
    ''' 
    
    def __init__(self, k=5, init_values=0):
        self.k=k
        self.init_values=init_values
        self.N_samples = np.zeros(self.k)
        
        ## create Values object to efficiently pass various parameters needed to evaluate value estimates
        class Values(object):
            pass
        self.values = Values()
        self.values.Q = self.init_values*np.ones(self.k)+np.random.normal(scale=0.01,size=self.k) # randomness to break ties in max fn.        
        self.values.N_samples = self.N_samples
            
    def reset(self):
        self.N_samples = np.zeros(self.k)
        self.values.Q = self.init_values*np.ones(self.k)+np.random.normal(scale=0.01,size=self.k) # randomness to break ties in max fn.        
        self.values.N_samples = self.N_samples
    
    def update_value(self, last_action, last_reward): 
        '''
        Update value estimates based on last reward.
        '''
        self.N_samples[last_action]+=1
        alpha = 1./self.N_samples[last_action]
        self.values.Q[last_action] += alpha*(last_reward-self.values.Q[last_action])
        self.values.N_samples = self.N_samples
        return self.values


class Agent_ThompsonSampling(Agent_SampleAverage):
    '''
    A collection of functions to implement Thompson Sampling value estimation.
    Works in conjuction with Policy_ThompsonSampling action policy, on a Bernoulli environment.
    Inherits Agent_SampleAverage class.
    ''' 
    
    def __init__(self, k=5):
        super(Agent_ThompsonSampling, self).__init__(k)
        self.values.a = np.ones(self.k)
        self.values.b = np.ones(self.k)
        self.values.Q = 0.5*np.ones(self.k)
    
    def reset(self):
        super(Agent_ThompsonSampling, self).reset()
        self.values.a = np.ones(self.k)
        self.values.b = np.ones(self.k)
        self.values.Q = 0.5*np.ones(self.k)
            
    def update_value(self, last_action, last_reward):
        '''
        Update value estimates based on last reward.
        '''
        self.N_samples[last_action]+=1

        if last_reward==1: 
            self.values.a[last_action]+=1
        else: 
            self.values.b[last_action]+=1      

        a, b = self.values.a[last_action], self.values.b[last_action]  
        self.values.Q[last_action] = a/(a+b)            
        self.values.N_samples = self.N_samples

        return self.values
    
    