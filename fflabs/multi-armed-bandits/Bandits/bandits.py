import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta
import pandas as pd
import seaborn as sns

class Bandit(object):    
    '''
    A collection of functions to build Bandit pipelines (environment, agent, policy), 
    run experiments, log results, and plot.
    ''' 

    def __init__(self, env=None, agent=None, policy=None):
        '''
        Build pipeline, initialize dummy arrays for logging individual runs.
        '''
        self.env    = env
        self.agent  = agent
        self.policy = policy
        self.k      = self.env.k
        self.action_arr, self.reward_arr, self.values_arr, self.labels_arr, self.N_pulls_arr = [], [], [], [], []

    def init_history(self, Nsteps=1000): 
        '''
        Initialize experiment history arrays.
        '''
        self.action_history = np.zeros(Nsteps)
        self.reward_history = np.zeros(Nsteps)
        self.values_history = np.zeros([self.k, Nsteps]) 
        self.opt_action_history = np.zeros(Nsteps)
        self.N_pulls = np.zeros(self.k) 

    def log_step(self, step):
        '''
        Log individual step.
        '''
        self.action_history[step] = self.policy.last_action
        self.reward_history[step] = self.env.last_reward
        self.values_history[:,step] = self.agent.values.Q
        self.opt_action_history[step] = int(self.policy.last_action == np.argmax(self.env.arm_centers))
        self.N_pulls[self.policy.last_action] += 1

    def log_avg_experiment(self, experiment_label = ''):
        '''
        Log individual experiment
        '''
        self.action_arr.append(self.action_bar)
        self.reward_arr.append(self.reward_bar)
        self.values_arr.append(self.values_bar)        
        self.N_pulls_arr.append(self.N_pulls_bar)
        self.labels_arr.append(experiment_label) 

    def run_experiment(self, Nsteps=1000):
        '''
        Run invidiual experiment
        '''
        self.agent.reset()    
        values = self.agent.values
        self.init_history(Nsteps)
        for step in range(Nsteps):
            action = self.policy.select_action(values)
            reward = self.env.get_reward(action) 
            values = self.agent.update_value(action, reward)
            self.log_step(step)        

    def avg_experiment(self, Nruns=500, Nsteps=1000, experiment_label=None):
        '''
        Run multiple experiments and average
        '''
        import tqdm
        self.action_bar, self.reward_bar, self.values_bar, self.N_pulls_bar = np.zeros(Nsteps), np.zeros(Nsteps), np.zeros([self.k, Nsteps]) , np.zeros(self.k)
        for _ in tqdm.tqdm(range(Nruns)):
            self.run_experiment(Nsteps=Nsteps)
            self.action_bar += self.opt_action_history/Nruns
            self.reward_bar += self.reward_history/Nruns
            self.values_bar += self.values_history/Nruns
            self.N_pulls_bar+= self.N_pulls/Nruns
        self.log_avg_experiment(experiment_label)

    def plot(self, keys=['action']):
        '''
        Plot average experiments. Use variable list "keys" to specify which plots to create.
        keys: 'reward', 'action', 'values', 'regret'
        corresponding to: reward history, optimal action history, value estimate history, and cumulative action selection rate table.
        '''

        Nsteps = len(self.action_arr[0])
        
        ## Setup figure      
        fig, axs = plt.subplots(1,len(keys), figsize=(16,5)) 
        for i,key in enumerate(keys):
            if len(keys) == 1: ax = axs
            else: ax = axs[i]
            ## Figure out what things to plot
            if key == 'regret':
                self.draw_regret_table(ax)              
            else:
                for action, reward, values, label in zip(self.action_arr, self.reward_arr, self.values_arr, self.labels_arr):                    
                    if key == 'reward':                
                        ax.plot(range(0,Nsteps), reward, alpha=0.7, label=label); 
                        ax.set_ylabel('Reward', fontsize=14) 
                    elif key =='action':
                        ax.plot(range(0,Nsteps), action, alpha=0.7, label=label); 
                        ax.set_ylabel('Fraction optimal action', fontsize=14) 
                    elif key == 'values':
                        for ii,values_k in enumerate(values):
                            ax.plot(range(0,Nsteps), values_k, alpha=0.9, linewidth=4, label='action = {}'.format(ii+1)); 
                        ax.set_ylabel('Value', fontsize=14)
                        ax.legend(prop={'size':14}) 
                    ax.legend(prop={'size':14})
                    ax.set_xlabel('Step',fontsize=14)
    
    def draw_regret_table(self, ax):
        '''
        Draw cumulative action selection rate table
        '''    
        action_labels = ["Action {}\n Q = {:4.2f}".format(x, y) for x, y in zip(range(1,6), self.env.arm_centers)]
        df = pd.DataFrame([b*100/len(self.action_history) for b in self.N_pulls_arr])
        ax = sns.heatmap(df, annot=True, ax=ax, annot_kws={"size":15}, cbar=False, fmt = '.2f', linewidth=0.5)
        for t in ax.texts: t.set_text(t.get_text() + " %")
        ax.set_yticklabels(self.labels_arr[::-1], rotation=0,fontsize=13);
        ax.set_xticklabels(action_labels,fontsize=13)
        ax.set_title('Cumulative Action Selection Rate',fontsize=13)

class Bandit_Bernoulli(Bandit):
    '''
    A collection of functions to build Bernoulli Bandit pipelines (environment, agent, policy), 
    run experiments, log results, and plot. 
    Inherits Bandit class functions, updating some of the logging functions.
    ''' 

    def init_history(self, Nsteps=1000):
        '''
        Initialize experiment history arrays. Inherits Bandit init_history() function.
        Adds arrays for beta parameters alpha and beta.
        '''
        super(Bandit_Bernoulli,self).init_history(Nsteps)
        self.a_history = np.zeros([self.k, Nsteps])
        self.b_history = np.zeros([self.k, Nsteps])
        
    def log_step(self, step):
        '''
        Log individual step. Inherits Bandit log_step() function.
        Adds arrays for beta parameters alpha and beta.
        '''
        super(Bandit_Bernoulli,self).log_step(step)
        self.a_history[:,step] = self.agent.values.a
        self.b_history[:,step] = self.agent.values.b
        
    def plot_beta(self, plot_steps):
        '''
        Visualize representative Beta distributions in a single experimental run.
        '''
        sns.set(font_scale=1)
        nrows=len(plot_steps)/2
        fig, ax = plt.subplots(nrows=2,ncols=nrows,figsize=(18,7))
        x=np.linspace(0,0.2,200)
        for i, step in enumerate(plot_steps):
            A = self.a_history[:,step]
            B = self.b_history[:,step]
            ax = plt.subplot(2, nrows, i+1)
            for a,b,p in zip(A,B,self.env.arm_centers):
                rv = beta(a, b)
                ax.plot(x, rv.pdf(x), '-', lw=2, label='p = {:0.2f}'.format(p));
                ax.fill_between(x, np.zeros(len(x)), rv.pdf(x), alpha=0.1);
                ax.set_yticks([])
            ax.set_title('Step = {}'.format(step))
            ax.legend()