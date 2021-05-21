import torch
import numpy as np
import matplotlib.pyplot as plt

# set up matplotlib
plt.ion()


# Helper function to plot duration of episodes and an average over the last 100 iterations
def plot_variables(fig, plot_var, title):
    fig, ax1, ax2 = fig
    durations_t = torch.tensor(plot_var['durations'], dtype=torch.float)
    rewards_t = torch.tensor(plot_var['rewards'], dtype=torch.float)
    

    fig.suptitle(title)
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Duration')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Reward')
    ax1.plot(durations_t.numpy(), 'bo')
    ax2.plot(rewards_t.numpy(), 'bo')

    # Take 100 episode averages and plot them too
    if len(durations_t) >= 50:
        means = durations_t.unfold(0, 50, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(49), means))
        ax1.plot(means.numpy(), 'r')
        means = rewards_t.unfold(0, 50, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(49), means))
        ax2.plot(means.numpy(), 'r')

    plt.pause(0.001)    # Small pause to update plots

