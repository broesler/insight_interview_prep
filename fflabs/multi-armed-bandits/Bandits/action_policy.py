# Agent Action policy
import numpy as np
from scipy.stats import beta

class Policy_EpsilonGreedy(object):
    '''
    A collection of functions to implement an epsilon-greedy action policy.
    ''' 
    
    def __init__(self, k=5, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        
    def select_action(self, values):
        '''
        Action selection based on current value estimates.
        '''
        r = np.random.random()
        if r<self.epsilon: 
            self.last_action = np.random.choice(self.k)
        else: 
            self.last_action = np.argmax(values.Q)   

        return self.last_action

class Policy_Softmax(Policy_EpsilonGreedy):
    '''
    A collection of functions to implement a softmax action policy. 
    Inherits Policy_EpsilonGreedy class.
    ''' 

    def __init__(self, k=5, init_values=0, tau=0.1):
        super(Policy_Softmax,self).__init__(k)
        self.tau=tau
                
    def select_action(self, values):    
        '''
        Action selection based on current value estimates.
        '''    
        PSM_arg = np.exp(values.Q/self.tau)
        PSM = PSM_arg / sum(PSM_arg)
        self.last_action = np.random.choice(a=self.k, p=PSM)

        return self.last_action
    

class Policy_ThompsonSampling(object):
    '''
    A collection of functions to implement Thompson Sampling. 
    Works in conjuction with Agent_ThompsonSampling value estimation, on a Bernoulli environment.
    ''' 
    
    def __init__(self, k=3):
        self.k = k
        
    def select_action(self, values):
        '''
        Action selection based on current value estimates.
        '''
        samples = [np.random.beta(a,b) for a,b in zip(values.a, values.b)]
        self.last_action = np.argmax(samples)
        return self.last_action

class Policy_UCB(object):
    '''
    A collection of functions to implement an Upper Confidence Bound action policy (UCB1). 
    ''' 
    
    def __init__(self, k=3, c=2.0, init_exploration = 1):
        self.k = k
        self.c = c
        self.init_exploration = init_exploration
        
    def select_action(self, values):
        '''
        Action selection based on current value estimates.
        '''
        t = np.sum(values.N_samples)
        with np.errstate(divide='ignore', invalid='ignore'):          
          exploration = np.log(t+1) / values.N_samples
        exploration[np.isinf(exploration)] = self.init_exploration
        exploration = np.power(exploration, 1/self.c)

        UCB = values.Q + exploration
        self.last_action = np.argmax(UCB)
        
        return self.last_action