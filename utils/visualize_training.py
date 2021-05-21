import numpy as np
import pickle
import argparse
import scipy.ndimage as sciimg
from sliding_window import sliding_window_view
import matplotlib.pyplot as plt


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

def plot_variables(args, figure, plot_var):
    plt_title = args.title
    win_size = args.win_size
    reward_only = args.reward_only
    if reward_only:
        fig, ax = figure
    else:
        fig, ax1, ax2 = figure
    durations = np.array(plot_var['durations'])
    rewards = np.array(plot_var['rewards'])

    dur_expand = sliding_window_view(durations, [win_size], steps=[1])
    rew_expand = sliding_window_view(rewards, [win_size], steps=[1])

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

    if reward_only:
        fig.suptitle(plt_title)
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.plot(x, rm, '#30BFBF')
        ax.fill_between(x, rlb, rub, alpha=1, edgecolor='#A7DADF', facecolor='#C4E1E7')
        ax.set_xlim(0, rewards.shape[0])
        ax.set_ylim(0, 216)
        ax.set_facecolor('#EAEAF2')
        ax.grid(color='w', b=True, which='major')
    else:
        fig.suptitle(plt_title)
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Duration')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Reward')
        ax1.plot(x, rm, 'b')
        ax1.fill_between(x, rlb, rub, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')
        ax2.plot(x, dm, 'b')
        ax2.fill_between(x, dlb, dub, alpha=0.5, edgecolor='#CC4F1B', facecolor='#FF9848')

def main(args):
    result_path = args.result_path

    with open(result_path, 'rb') as file:
        directory = pickle.load(file)

    # Prepare the drawing figure
    if args.reward_only:
        fig, ax = plt.subplots(1, 1)
        figure = (fig, ax)
    else:
        fig, (ax1, ax2) = plt.subplots(1, 2)
        figure = (fig, ax1, ax2)
    plot_variables(args, figure, directory)
    plt.show(block=True)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Plot training results with shaded confidence interval')
    parser.add_argument('--result_path', default='/home/jc/logs/haptics_pomdp_4_300_006_180/results_pomdp.pkl', help='result path')
    parser.add_argument('--title', default="No translation in initialization", help='plot title')
    parser.add_argument('--win_size', default=50, type=int, help='Window size for mean')
    parser.add_argument('--reward_only', default=True, type=bool, help='whether or not plot reward only or duration as well')
    parser.add_argument('--sigma', default=1.5, type=int, help='std for smooth data')

    args = parser.parse_args()
    
    main(args)

