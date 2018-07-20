# Environment
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy.stats import beta

class Env_Gaussian(object): 
    '''
    A collection of functions to implement and visualize a Gaussian environment.
    Rewards are drawn from a Gaussian distirubtion with unit variance.
    ''' 
    
    def __init__(self, k=5, seed=None):
        '''
        Initialize reward distributions
        '''
        self.k = k
        if seed != None:
            np.random.seed(seed)
        self.arm_centers = np.random.normal(loc=0, scale=1, size=self.k)
        self.arm_widths  = np.ones(self.k)

    def get_reward(self, last_action):
        '''
        Draw a reward for the corresponding action.
        '''
        self.last_reward = np.random.normal(loc=self.arm_centers[last_action], 
                                            scale=self.arm_widths[last_action], size=1)        
        return self.last_reward
    
    def sample_env(self):
        '''
        Draw a number of samples from each environment. Mainly used for visualization.
        '''
        N_samples = 10000
        self.samples = np.empty([N_samples,self.k])
        for i, center, width in zip(range(self.k), self.arm_centers, self.arm_widths):
            self.samples[:,i] = np.random.normal(loc=center, scale=width, size=N_samples)
        
    def visualize_env(self):
        '''
        Visualize Gaussian environment using violin plots.
        '''        
        self.sample_env()    
        df = pd.DataFrame(self.samples)
        fig, ax = plt.subplots(1,1, figsize=(16,4))
        ax = sns.violinplot(data=df)
        labels = ['Action {}\n {:4.2f}'.format(i,v) for i,v in zip(range(1,self.k+1), self.arm_centers)]        
        ax.set_xticklabels(labels,fontsize=13);
        ax.set_title("{}-Armed Bandit Gaussian Environemnt".format(self.k),fontsize=15);   

class Env_Bernoulli(Env_Gaussian):
    '''
    A collection of functions to implement and visualize a Bernoulli environment.
    Rewards are drawn from a Bernoulli distirubtion, parametrized by success probability p_k.
    Inherits Env_Gaussian class.
    '''     

    def __init__(self, k=3, seed=None, p=None):
        '''
        Initialize reward distributions. Inherits Env_Gaussian __init__().
        '''
        super(Env_Bernoulli,self).__init__(k=k, seed=seed)
        if p==None: 
            p = np.random.uniform(0,0.2, size=k)        
        self.arm_centers = p
    
    def get_reward(self, last_action):
        '''
        Draw a reward for the corresponding action.
        '''
        self.last_reward = np.random.binomial(1,self.arm_centers[last_action],1)    
        return self.last_reward
    
    def sample_env(self):
        '''
        Draw a number of samples from each environment. Mainly used for visualization.
        '''
        N_samples = 100
        self.samples = np.empty([N_samples,self.k])
        for i, p in zip(range(self.k), self.arm_centers):
            self.samples[:,i] = [np.mean(np.random.binomial(1, p ,1000)) for ii in range(N_samples)]

    def visualize_env(self):
        '''
        Visualize Bernoulli environment using violin plots. 
        This plots the distribution of estimated success probabilities given a series of experiments, each involving 100 samples.
        '''    
        self.sample_env()   
        N_beta_samples = 10000
        beta_samples = np.empty([N_beta_samples,self.k]) 
        for i, sample in enumerate(np.transpose(self.samples)):            
            a = np.sum(sample)
            b = len(sample) - a
            beta_samples[:,i] = [np.random.beta(a,b) for _ in range(N_beta_samples)]
        df = pd.DataFrame(beta_samples)        
        fig, ax = plt.subplots(1,1, figsize=(16,3))
        ax = sns.violinplot(data=df)
        labels = ['Action {}\n{:0.2f}'.format(i,v) for i,v in zip(range(1,self.k+1),self.arm_centers)]
        ax.set_xticklabels(labels,fontsize=13);
        ax.set_title("{}-Armed Bandit Bernoulli Environment".format(self.k),fontsize=18);   
