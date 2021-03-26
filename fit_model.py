from enviroment import reduced_two_step_env
from agents import binary_cross_entropy
from agents import model_free_with_eligibility_trace_agent
from agents import model_based_agent_change_point, binary_cross_entropy
from visualization import plot_model, plot_behaviour

from data_loading import load_all_data 
import copy
dcp = copy.deepcopy

import numpy as np
import scipy as sp

data = load_all_data('Experiment')
#data = load_all_data('Experiment')
subjects = list(data.keys())
subject_name = subjects[0]

# stuff for plot
indexes = []
actions_mice = []
actions_agent = []
optimal_actions = []
Qs = []
Ts = []
session_tag = {}
block_map = {'U': 0, 'D':1, 'N': 0.5}
inspect_model = False

num_states=3
num_actions=3

def entropy_train(v):
    # model parameters
    alpha=lambda x: v[0]
    beta=lambda x: 3
    gamma=lambda x: 0.99#v[1]
    lamda=lambda x: 0.99#v[2]
    H_lamda = 75
    agent = model_free_with_eligibility_trace_agent(num_states, num_actions, alpha, beta, gamma, lamda)
    #agent = model_based_agent_change_point(num_states, num_actions, beta, H_lamda)
    return single_run(agent)

def single_run(agent):
    Hs = []
    plot_flag = False
    idx = 0
    for s in range(len(data[subject_name])):
        data_session = data[subject_name][s]
        rl_data_session = data_session['state_transition']
        if data_session['training_stage'] == '4.7': plot_flag = True
        #if data_session['training_stage'] == '4.7': break
        session_tag[idx] = data_session['training_stage']
        for t in range(len(rl_data_session)):
            data_trial = rl_data_session[t]
            choice_type = data_session['choice_type'][t]
            block_type = block_map[data_session['block_type'][t][0]]
            for i in range(len(data_trial)):
                dat = data_trial[i]
                state, action, reward, new_state = dat[0], dat[1], dat[2], dat[3]
                agent.update(state, action, reward, new_state)
                if state is 0: 
                    idx+=1
                    #if choice_type == 'FC':
                    indexes.append(idx)
                    actions_mice.append(action)
                    prob_act_1 = agent.softmax(agent.Q[0, 0:2]*3)[1]
                    actions_agent.append(prob_act_1)
                    optimal_actions.append(block_type)
                    # cross entropy calculation for free choice trails
                    cross_entropy = binary_cross_entropy(action, agent.softmax(agent.Q[0, 0:2])) 
                    Hs.append(cross_entropy)
                    Qs.append(dcp(agent.Q))
                    # calculate model params
                    if inspect_model:
                        agent.calculate_model_params()
                        Ts.append(dcp(agent.theta_mus))
            #if plot_flag is True: Qs.append(dcp(agent.Q))
            agent.reset()
    print(sum(Hs))
    return sum(Hs)


def optimize_params():
    from scipy import optimize
    #bounds = (slice(0,1.0,0.1), slice(0,1.0,0.1), slice(0,1.0,0.1)) 
    bounds = (slice(0, 1., 0.1), slice(0,1.0,0.1))#, slice(0,1.0,0.1)) 
    results = optimize.brute(entropy_train, bounds) 
    print(results)
    print(results.x)


if __name__ == "__main__":
    entropy_train([0.46])
    #optimize_params()
    #plot_model(Qs, session_tag, plot_type='params')
    #plot_model(Ts, session_tag, plot_type='params')
    plot_behaviour(indexes, actions_mice, actions_agent, optimal_actions, session_tag, window_size=5)


