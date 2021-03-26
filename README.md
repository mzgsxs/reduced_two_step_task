# reduced_two_step_task

data_extraction script loads the raw data from pycontrol log files and saves as pickle files.
data_loading script loads pickle files save by data_extraction.

enviroment, agent, defines reinforcement learning framwork and the agents that interacts with.
two agent has been defined:
1. model-free agent with eligibility trace.
2. bayesian change point detection model. It consist of four independet parameters/factors - transition probability θ_left from state S_0 to S_up for poke S_left (so 1 - θ_left fot probability of transit to S_down, and similarly for θ_right and S_right), reward_probability θ_up for receiving reward after transited and poked S_up(so 1 - θ_up for probability of no reward, similarly for θ_down) - are the parameters of the bernulli model, and a hyperfactor - "running length" - controls the length of the statistics used for bayesian updating. This model deploy a full bayesian updating to update all four θ factos and running length factor r.

fit_model script uses data_loading, enviroment and agent to fit the data to one specified model, cross entropy is calculate for the action of the mice and the score of the softmax policy.

visulization script plots:
1. the model values/parameters changes though time and sessions.
2. comparasion of the optimal action, mice action(moving averaged smoothed over n trails) and agent softmax score.
