import numpy as np
import copy

def binary_cross_entropy(P, Q):
    '''
    binary variable with support of {0, 1}
    Input;
        P: actual action performed, one hot index
        Q: soft score
    Output:
        H: entropy 
    '''
    return -np.log(Q)[P] 


 
class model_free_with_eligibility_trace_agent():
    def __init__(self, num_states, num_actions, alpha, beta, gamma, lamda):
        print("model free agent with eligibility trace has been initiated")
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.lamda = lamda
        self.eligibility_trace = np.zeros((num_states, num_actions))
        self.Q = np.zeros((num_states, num_actions))
        self.score = np.zeros((num_actions))
        self.td_error = 0.
        
    def softmax(self, q_s):
        # softmax 
        exp_q_state = np.exp(q_s)
        score = exp_q_state/np.sum(exp_q_state)
        return score
    
    def act(self, state):
        # softmax policy
        if state is 0:
            q_s = self.Q[state,0:2]*self.beta(0)
            score = self.softmax(q_s)
            action = np.random.multinomial(1,score)
            action = np.nonzero(action)[0]
        else:
            action = 2
        return int(action)
        
    def update(self, state, action, reward, new_state):
        self.eligibility_trace[state,action] += 1
        if new_state>0:# if new state is not terminal
            V_new_state = self.Q[new_state,2]
        else:
            V_new_state = 0
        self.td_error = reward + self.gamma(0)*V_new_state - self.Q[state, action]
        self.Q += self.alpha(0)*self.td_error*self.eligibility_trace
        self.eligibility_trace *= self.gamma(0)*self.lamda(0)
    
    def reset(self):
        self.eligibility_trace = np.zeros((self.num_states, self.num_actions))



class model_based_agent_change_point():
    def __init__(self, num_states, num_actions, beta, H_lamda):
        print("model based agent with change point detection and eligibility trace has been initiated")
        self.Q = np.ones((num_states, num_actions))*0.5
        self.theta_mus = np.zeros((num_states, num_actions))
        self.alpha = np.zeros((num_states, num_actions))
        self.beta = beta
        # bernuli model 
        # hyper-parameter alpha, beta of 
        # harzard function with geometric distribution 
        # with initial run length at 0 with probability 1
        self.H = lambda tau: 1./H_lamda
        #self.r_distribution = [[[1] for j in range(num_actions)] for i in range(num_states)]
        self.r_distribution = [[{0:1} for j in range(num_actions)] for i in range(num_states)]
        self.statistics = [[[] for j in range(num_actions)] for i in range(num_states)]
        self.action_last = 2

    def softmax(self, q_s):
        # softmax 
        exp_q_state = np.exp(q_s)
        score = exp_q_state/np.sum(exp_q_state)
        return score
    
    def act(self, state):
        # softmax policy
        if state is 0:
            q_s = self.Q[state,0:2]*self.beta(0)
            score = self.softmax(q_s)
            action = np.random.multinomial(1,score)
            action = np.nonzero(action)[0]
        else:
            action = 2
        return int(action)
 
    def calculate_model_params(self):
        for state, action in [(0,0),(0,1),(1,2),(2,2)]:
          self.theta_mus[state, action] = self._get_model_param(state, action)
    
    def _get_model_param(self, state, action):
        theta_mu_given_r = 0.
        for r in self._run_length_list(state, action):
            beta = 1 if r is 0 else sum(self.statistics[state][action][-r:]) + 1 
            alpha = 1 if r is 0 else r - beta + 2
            theta_mu = alpha/(alpha+beta)
            theta_mu_given_r += self.r_distribution[state][action][r]*theta_mu
        return theta_mu_given_r        
 
    def _run_length_list(self, state, action):
        return sorted(list(self.r_distribution[state][action].keys()))

    def _update(self, state, action, reward, new_state, num_r):
        transited_state = new_state-1 if state is 0 else 1-reward
        r_distribution_new = {0:0.}
        
        for r in self._run_length_list(state, action):
            prob_reset = self.H(r)
            prob_grow = 1. - prob_reset
            r_distribution_new[0] += self.r_distribution[state][action][r]*prob_reset
            r_distribution_new[r+1] = self.r_distribution[state][action][r]*prob_grow
            
        V_g = self.Q[1,2] if state is 0 else 1.
        V_b = self.Q[2,2] if state is 0 else 0.
        # posteior distribution of run length
        r_list_new = sorted(list(r_distribution_new.keys()))
        r_x_distribution = np.zeros((num_r+1,2))
        for i, r in enumerate(r_list_new):
            beta = 1 if r is 0 else sum(self.statistics[state][action][-r:]) + 1 
            alpha = 1 if r is 0 else len(self.statistics[state][action][-r:]) - beta + 1
            theta_mu = (alpha*V_g+beta*V_b)/(alpha+beta)
            r_x_distribution[i,0] = r_distribution_new[r]*theta_mu
            r_x_distribution[i,1] = r_distribution_new[r]*(1-theta_mu)
        r_given_x_distribution = r_x_distribution/np.sum(r_x_distribution,0)
        r_posterior = {}
        for i, r in enumerate(r_list_new):
            r_prob = r_given_x_distribution[i,int(transited_state)]
            if r_prob > 0.01: r_posterior[r] = r_prob 
        self.r_distribution[state][action] = r_posterior 

        # prediction of observation 
        Q = 0.
        for r in self._run_length_list(state, action):
            beta = 1 if r is 0 else sum(self.statistics[state][action][-r:]) + 1 
            alpha = 1 if r is 0 else r - beta + 2
            theta_mu = (alpha*V_g+beta*V_b)/(alpha+beta)
            Q += self.r_distribution[state][action][r]*theta_mu
        
        Q_old = self.Q[state,action]
        Q_next = self.Q[new_state,2] if state is 0 else reward
        td_error = Q_next - Q_old
        alpha = (Q - Q_old)/td_error if abs(td_error) > 0 else 0
        self.Q_old = copy.deepcopy(self.Q)
        self.Q[state,action] = Q
        self.alpha[state,action] = alpha
        
    def update(self, state, action, reward, new_state):
        # update statistics
        transited_state = new_state-1 if state is 0 else 1-reward
        self.statistics[state][action].append(transited_state)

        # prior distribution of run length
        num_r = len(self.r_distribution[state][action].keys())
        # update twice of state is final
        self._update(state, action, reward, new_state, num_r)
        if state is not 0:
            # prior distribution of run length
            num_r = len(self.r_distribution[0][self.action_last].keys())
            self._update( 0, self.action_last, 0, state, num_r)
        self.action_last = action
        
    def reset(self):
        pass 



'''
class model_based_agent_change_point():
    def __init__(self, num_states, num_actions, beta, H_lamda):
        print("model based agent with change point detection and eligibility trace has been initiated")
        self.Q = np.ones((num_states, num_actions))*0.5
        self.alpha = np.zeros((num_states, num_actions))
        self.beta = beta
        # bernuli model 
        # hyper-parameter alpha, beta of 
        # harzard function with geometric distribution 
        # with initial run length at 0 with probability 1
        self.H = lambda tau: 1./H_lamda
        self.r_distribution = [[[1] for j in range(num_actions)] for i in range(num_states)]
        self.statistics = [[[] for j in range(num_actions)] for i in range(num_states)]
        self.action_last = 2

    def softmax(self, q_s):
        # softmax 
        exp_q_state = np.exp(q_s)
        score = exp_q_state/np.sum(exp_q_state)
        return score
    
    def act(self, state):
        # softmax policy
        if state is 0:
            q_s = self.Q[state,0:2]*self.beta(0)
            score = self.softmax(q_s)
            action = np.random.multinomial(1,score)
            action = np.nonzero(action)[0]
        else:
            action = 2
        return int(action)
    
    def _update(self, state, action, reward, new_state, max_r):
        transited_state = new_state-1 if state is 0 else 1-reward
        r_distribution_new = [0.]*(max_r+1)
        
        for r in range(max_r):
            prob_reset = self.H(r)
            prob_grow = 1. - prob_reset
            r_distribution_new[0] += self.r_distribution[state][action][r]*prob_reset
            r_distribution_new[r+1] += self.r_distribution[state][action][r]*prob_grow
            
        V_g = self.Q[1,2] if state is 0 else 1.
        V_b = self.Q[2,2] if state is 0 else 0.
        # posteior distribution of run length
        r_x_distribution = np.zeros((max_r+1,2))
        for r in range(max_r+1):
            beta = 1 if r is 0 else sum(self.statistics[state][action][-r:]) + 1 
            alpha = 1 if r is 0 else len(self.statistics[state][action][-r:]) - beta + 1
            theta_mu = (alpha*V_g+beta*V_b)/(alpha+beta)
            r_x_distribution[r,0] = r_distribution_new[r]*theta_mu
            r_x_distribution[r,1] = r_distribution_new[r]*(1-theta_mu)
        r_given_x_distribution = r_x_distribution/np.sum(r_x_distribution,0)
        self.r_distribution[state][action] = r_given_x_distribution[:,int(transited_state)]

        # prediction of observation 
        Q = 0.
        for r in range(max_r+1):
            beta = 1 if r is 0 else sum(self.statistics[state][action][-r:]) + 1 
            alpha = 1 if r is 0 else r - beta + 2
            theta_mu = (alpha*V_g+beta*V_b)/(alpha+beta)
            Q += self.r_distribution[state][action][r]*theta_mu
        
        Q_old = self.Q[state,action]
        Q_next = self.Q[new_state,2] if state is 0 else reward
        td_error = Q_next - Q_old
        alpha = (Q - Q_old)/td_error if abs(td_error) > 0 else 0
        self.Q_old = copy.deepcopy(self.Q)
        self.Q[state,action] = Q
        self.alpha[state,action] = alpha
        
    def update(self, state, action, reward, new_state):
        # update statistics
        transited_state = new_state-1 if state is 0 else 1-reward
        self.statistics[state][action].append(transited_state)

        # prior distribution of run length
        max_r = len(self.r_distribution[state][action])
       
        # update twice of state is final
        self._update(state, action, reward, new_state, max_r)
        if state is not 0:
            # prior distribution of run length
            max_r = len(self.r_distribution[0][self.action_last])
            self._update( 0, self.action_last, 0, state, max_r-1)
        self.action_last = action
        
    def reset(self):
        pass 
'''
