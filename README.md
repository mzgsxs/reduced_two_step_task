# reduced_two_step_task

data_extraction script loads the raw data from pycontrol log files and saves as pickle files.
data_loading script loads pickle files saved by data_extraction.
data folder should be placed at the root directory and organized as the following:
- ROOT/data/BATCH#/SUBJECT#/Training
- ROOT/data/BATCH#/SUBJECT#/Experiment

BATCH# is the batch number, for example "2019.12-2020.01", SUBJECT# is the subject number of the mice, for example "Two_step_WT1". It will save pickle files into:
- ROOT/pickle/BATCH#/Training/file.pickle
- ROOT/pickle/BATCH#/Experiment/file.pickle

All sucjects in the same batch will be saved into the same file.pickle. This file.pickle when loaded is a python dictionary as:
```json
{SUBJECT_NAME:{'training_stage':STAGE,
               ...
               'state_transition':[{0:(state, action, reward, new_state), 1:()}, ...],
               'choice_type':['UA', ...],
               'block_type':[]
               }
```

enviroment, agent, defines reinforcement learning framwork and the agents that interacts with.
two agent has been defined:
1. model-free agent with eligibility trace.
2. bayesian change point detection model. It consist of four independet parameters/factors - transition probability θ_left from state S_0 to S_up for poke S_left (so 1 - θ_left fot probability of transit to S_down, and similarly for θ_right and S_right), reward_probability θ_up for receiving reward after transited and poked S_up(so 1 - θ_up for probability of no reward, similarly for θ_down) - are the parameters of the bernulli model, and a hyperfactor - "running length" - controls the length of the statistics used for bayesian updating. This model deploy a full bayesian updating to update all four θ factos and running length factor r.

fit_model script uses data_loading, enviroment and agent to fit the data to one specified model, cross entropy is calculate for the action of the mice and the score of the softmax policy.

visulization script plots:
1. the model values/parameters changes though time and sessions.
2. comparasion of the optimal action, mice action(moving averaged smoothed over n trails) and agent softmax score.
