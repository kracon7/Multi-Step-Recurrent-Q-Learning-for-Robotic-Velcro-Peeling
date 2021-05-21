import numpy as np
import pickle
import argparse
import scipy.ndimage as sciimg
from sliding_window import sliding_window_view
import matplotlib.pyplot as plt

plt.ion()

def var_of_means(window, win_size, N):
    means = []
    for i in range(N):
        sample = []
        for j in range(win_size):
            sample.append(np.random.choice(window))
        means.append(np.average(sample))
    return np.average(window), np.std(means)

def smooth_array(array_list, sigma=5):
    result = []
    for item in array_list:
        result.append(sciimg.gaussian_filter(item, sigma=sigma, mode='reflect'))
    return result


def main(args):
    win_size = args.win_size
    plt_title = args.title
    result_path = '/home/jc/logs/triple_dqn/results_geom_mdp.pkl'
    with open(result_path, 'rb') as file:
        plot_var = pickle.load(file)
    durations_1 = np.array(plot_var['durations'])
    rewards_1 = np.array(plot_var['rewards'])

    N = rewards_1.shape[0]

    result_path = '/home/jc/logs/geom_mdp_2/result_geom_mdp.pkl'
    with open(result_path, 'rb') as file:
        plot_var = pickle.load(file)
    durations_2 = np.array(plot_var['durations'])
    rewards_2 = np.array(plot_var['rewards'])
    durations_2 = durations_2[:N] * 30/18
    rewards_2 = rewards_2[:N]

    fig, ax = plt.subplots(1, 1)
    figure = (fig, ax)

    dur_expand = sliding_window_view(durations_1, [win_size], steps=[1])
    rew_expand = sliding_window_view(rewards_1, [win_size], steps=[1])

    M = dur_expand.shape[0]
    rm = np.empty(M)
    dm = np.empty(M)
    rub = np.empty(M)
    rlb = np.empty(M)
    dub = np.empty(M)
    dlb = np.empty(M)

    for i in range(M):
        mean_r, var_r = var_of_means(rew_expand[i,:], win_size, 20)
        mean_d, var_d = var_of_means(dur_expand[i,:], win_size, 20)
        rm[i] = mean_r
        dm[i] = mean_d
        rub[i] = mean_r + 1.96 * var_r
        rlb[i] = mean_r - 1.96 * var_r
        dub[i] = mean_d + 1.96 * var_d
        dlb[i] = mean_d - 1.96 * var_d

    # further smooth the data
    [rm, dm, rub, rlb, dub, dlb] = smooth_array([rm, dm, rub, rlb, dub, dlb], sigma = args.sigma)

    x = np.arange(win_size, win_size + M)

    fig.suptitle(plt_title)
    ax.set_xlabel('Episode')
    ax.set_ylabel('Rewards')
    ax.plot(x, rm, '#30BFBF')
    ax.fill_between(x, rlb, rub, alpha=1, edgecolor='#A7DADF', facecolor='#C4E1E7')
    ax.set_xlim(0, rewards_1.shape[0])
    ax.set_ylim(0, 230)
    ax.set_facecolor('#EAEAF2')
    ax.grid(color='w', b=True, which='major')
######################################################################################################
    dur_expand = sliding_window_view(durations_2, [win_size], steps=[1])
    rew_expand = sliding_window_view(rewards_2, [win_size], steps=[1])

    M = dur_expand.shape[0]
    rm = np.empty(M)
    dm = np.empty(M)
    rub = np.empty(M)
    rlb = np.empty(M)
    dub = np.empty(M)
    dlb = np.empty(M)

    for i in range(M):
        mean_r, var_r = var_of_means(rew_expand[i,:], win_size, 20)
        mean_d, var_d = var_of_means(dur_expand[i,:], win_size, 20)
        rm[i] = mean_r
        dm[i] = mean_d
        rub[i] = mean_r + 1.96 * var_r
        rlb[i] = mean_r - 1.96 * var_r
        dub[i] = mean_d + 1.96 * var_d
        dlb[i] = mean_d - 1.96 * var_d

    # further smooth the data
    [rm, dm, rub, rlb, dub, dlb] = smooth_array([rm, dm, rub, rlb, dub, dlb], sigma = args.sigma)

    fig.suptitle(plt_title)
    ax.plot(x, rm, '#2171b4')
    ax.fill_between(x, rlb, rub, alpha=1, edgecolor='#6badd5', facecolor='#daebf5')

    plt.show(block=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot training results with shaded confidence interval')
    parser.add_argument('--title', default="Total Rewards: \nClipped DDQN vs Fixed step updated DDQN", help='plot title')
    parser.add_argument('--win_size', default=20, type=int, help='Window size for mean')
    parser.add_argument('--sigma', default=1.5, type=int, help='std for smooth data')

    args = parser.parse_args()
    
    main(args)

