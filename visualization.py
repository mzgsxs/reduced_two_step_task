import matplotlib
import matplotlib.pyplot as plt


def plot_model(Qs, session_tag, plot_type='value'):
    if plot_type is 'value':
        labels = [['Q(S_0, A)', 'Q(S_0, B)'], ['V(S_Good)', 'V(S_Bad)']]
    elif plot_type == 'params':
        labels = [['theta(S_0, A)', 'theta(S_0, B)'], ['theta(S_Good)', 'theta(S_Bad)']]
    else:
        print("something is wrong, can not decide the label.")
    indexs = [[[0,0],[0,1]],[[1,2],[2,2]]]
    fig, axs = plt.subplots(2, 2)
    for i in range(2):
        for j in range(2):
            idx = indexs[i][j]
            x = [Q[idx[0],idx[1]] for Q in Qs]
            ax = axs[i,j]
            ax.plot(range(1,len(x)+1), x, label=labels[i][j])
            ax.scatter(range(1,len(x)+1), x)
            tick_idxes, tick_labels = [], []
            for k in list(session_tag.keys()):
                ax.vlines(k,0,1,linestyles='dashed',color='r')
                tick_idxes.append(k), tick_labels.append(session_tag[k])
            ax.legend()
    plt.setp(axs, xticks=tick_idxes, xticklabels=tick_labels)
    plt.show()


def plot_behaviour(indexes, actions_mice, actions_agent, optimal_actions, session_tag, window_size=5):
    fig, ax = plt.subplots()
    x = actions_mice
    x2 = actions_agent
    # smoothing
    y = [sum(x[max(0,-window_size+i):i])/window_size for i in range(len(x))]
    y2 = [sum(x2[max(0,-window_size+i):i])/window_size for i in range(len(x2))]
    # plotting
    ax.plot(indexes, y, label="mice 5 trail average")
    ax.plot(indexes, x2, label="agent 5 trail average")
    ax.plot(indexes, optimal_actions, label="optimal rewarding choice")
    tick_idxes, tick_labels = [], []
    for k in list(session_tag.keys()):
        ax.vlines(k,0,1,linestyles='dashed',color='r')
        tick_idxes.append(k), tick_labels.append(session_tag[k])
    #ax.plot(indexes, Hs, label="5 trail average")
    ax.scatter(indexes, x)
    plt.xticks(tick_idxes, tick_labels)
    plt.yticks([0,1],['A','B'])
    ax.legend()
    plt.show()


