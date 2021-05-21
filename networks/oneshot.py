import os
import sys
import argparse
import random
import pickle
import time
from itertools import count
import numpy as np
from statistics import mean

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from mujoco_py import MjViewer, load_model_from_path, MjSim
import matplotlib.pyplot as plt
from networks.imitation_net import ImNet
from networks.dqn import Geom_DQN
from networks.conv_net import ConvNet
from networks.multimodal_pomdp import POMDP
from robot_sim import RobotSim
from sim_param import SimParameter

from utils.normalize import Normalizer, Geom_Normalizer, Multimodal_Normalizer
from utils.action_buffer import ActionSpace, TactileObs
from utils.velcro_utils import VelcroUtil
from utils.memory import ExpertMemory
from utils.gripper_util import init_for_test, norm_img, norm_depth


# Class that uses a Deep Recurrent Q-Network to optimize a POMDP
# The assumption here is that the observation o is equal to state s
class Oneshot:
    def __init__(self, args):
        self.args = args
        self.ACTIONS = ['left', 'right', 'forward', 'backward', 'up', 'down']  # 'open', 'close']
        self.P_START = 0.999
        self.P_END = 0.1
        self.P_DECAY = 60000
        self.max_iter = args.max_iter
        self.gripping_force = args.grip_force
        self.break_threshold = args.break_thresh

        # Prepare the drawing figure
        fig, axe = plt.subplots(1, 1)
        self.figure = (fig, axe)
        plt.ion()


    # Helper function to plot duration of episodes and an average over the last 100 iterations
    def plot_variables(self, fig, plot_var, title):
        fig, axe = fig
        loss = plot_var['loss']

        fig.suptitle(title)
        axe.set_xlabel('Episode')
        axe.set_ylabel('Loss')
        axe.set_yscale('log')
        axe.plot(loss.numpy(), 'bo')
        
        # Take 100 episode averages and plot them too
        if len(loss) >= 50:
            means = loss.unfold(0, 50, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(49), means))
            axe.plot(means.numpy(), 'r')
            
        plt.pause(0.001)    # Small pause to update plots

    def encode_label(self, labels):
        m, n, _ = labels.shape
        assert m == self.args.batch_size
        assert n == self.args.time_step
        onehot_labels = np.zeros((m,n,len(self.ACTIONS)))
        for i in range(m):
            for j in range(n):
                onehot_labels[i, j, labels[i,j]] = 1
        return onehot_labels


    def extract_ft(self, conv_net, obs_batch):
        args = self.args
        assert len(obs_batch) == args.batch_size
        assert len(obs_batch[0]) == args.time_step
        result = []
        for b in obs_batch:
            obs_squence = torch.tensor([]).to(args.device)
            for item in b:
                tactile_obs, img_norm = item
                torch_tactile_obs = torch.from_numpy(tactile_obs).float().to(args.device)
                torch_img_norm = torch.from_numpy(img_norm).float().to(args.device).unsqueeze(0)
                img_ft = conv_net.forward(torch_img_norm)
                torch_full_obs = torch.cat((torch_tactile_obs, img_ft[0]))
                obs_squence = torch.cat((obs_squence, torch_full_obs))
            torch_obs_squence = obs_squence.view(args.time_step, -1)
            result.append(torch_obs_squence)
        return torch.stack(result)

    def load_all(self):
        args = self.args
        # Create the output directory if it does not exist
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)

        self.policy_net = ImNet(args.policy_indim, args.outdim).to(args.device)
        self.policy_optimizer = optim.RMSprop(self.policy_net.parameters(), lr=args.lr)
        
        self.conv_net = ConvNet(args.ftdim, args.depth).to(args.device)
        self.conv_optimizer = optim.RMSprop(self.conv_net.parameters(), lr=args.lr)

        # Create policy_net and conv_net and optimizers
        if args.weight_policy:
            if os.path.exists(args.weight_policy):
                checkpoint = torch.load(args.weight_policy)
                self.policy_net.load_state_dict(checkpoint['policy_net'])
                self.policy_optimizer.load_state_dict(checkpoint['policy_optimizer'])

        if args.weight_conv:
            if os.path.exists(args.weight_conv):
                checkpoint = torch.load(args.weight_conv)
                self.conv_net.load_state_dict(checkpoint['conv_net'])
                self.conv_optimizer.load_state_dict(checkpoint['conv_optimizer'])

        # load expert trajectories
        self.memory = ExpertMemory(args.expert_traj)

    def set_lr(self):
        for g in self.conv_optimizer.param_groups:
            g['lr'] = self.args.lr
        for g in self.policy_optimizer.param_groups:
            g['lr'] = self.args.lr


    def train(self):
        args = self.args

        self.load_all()
        
        print_variable = {'loss': None}
        start_epoch = 0
        if args.weight_policy:
            if os.path.exists(args.weight_policy):
                checkpoint = torch.load(args.weight_policy)
                start_epoch = checkpoint['epoch']
                self.steps_done = start_epoch * args.num_steps
                with open(os.path.join(os.path.dirname(args.weight_policy), 'results_imitation.pkl'), 'rb') as file:
                    plot_dict = pickle.load(file)
                    print_variable['loss'] = plot_dict['loss']

        self.set_lr()

        action_space = ActionSpace(dp=0.06, df=10)
        BCELoss = nn.BCELoss(reduction='sum')
        
        for ii in range(start_epoch, start_epoch + args.epochs):
            start_time = time.time()
            epoch_loss = []

            # sample expert trajectories and labeled actions
            batch, labels = self.memory.sample(args.batch_size, args.time_step)

            # encode labels
            onehot_labels = self.encode_label(labels)

            torch_labels = torch.tensor(onehot_labels).float().to(args.device)

            for step in range(args.num_steps):
                # forward networks and predict action
                states = self.extract_ft(self.conv_net, batch)

                hidden_batch, cell_batch = self.policy_net.init_hidden_states(args.batch_size, args.device)

                prediction, h = self.policy_net.forward(states,
                                                 batch_size=args.batch_size,
                                                 time_step=args.time_step,
                                                 hidden_state=hidden_batch,
                                                 cell_state=cell_batch)

                ground_truth = torch_labels[:, args.time_step-1, :].squeeze(1)

                loss = BCELoss(prediction, ground_truth)

                # print and plot
                detached_loss = loss.detach().cpu()
                if print_variable['loss'] is not None:
                    print_variable['loss'] = torch.cat((print_variable['loss'], detached_loss.unsqueeze(0)))
                else:
                    print_variable['loss'] = detached_loss.unsqueeze(0)
                self.plot_variables(self.figure, print_variable, 'Training Imitation Agent')
                epoch_loss.append(detached_loss.item())

                # Optimize the model
                self.policy_optimizer.zero_grad()
                self.conv_optimizer.zero_grad()
                loss.backward()
                for param in self.policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                for param in self.conv_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                
                self.policy_optimizer.step()
                self.conv_optimizer.step()

            # end of training epoch    
            print("Epoch {} took {}s, average loss of this epoch: {}\n\n".format(ii, 
                                    time.time() - start_time, np.average(epoch_loss)))

            # Save checkpoints every vew iterations
            if ii % args.save_freq == 0:
                policy_save_path = os.path.join(args.output_dir, 'policy_checkpoint_' + str(ii) + '.pth')
                torch.save({
                           'epoch': ii,
                           'policy_net': self.policy_net.state_dict(),
                           'policy_optimizer': self.policy_optimizer.state_dict(),
                           }, policy_save_path)

                conv_save_path = os.path.join(args.output_dir, 'conv_checkpoint_' + str(ii) + '.pth')
                torch.save({
                           'epoch': ii,
                           'conv_net': self.conv_net.state_dict(),
                           'conv_optimizer': self.conv_optimizer.state_dict(),
                           }, conv_save_path)

        savefig_path = os.path.join(args.output_dir,'training_imitation')
        self.figure[0].savefig(savefig_path, format='png')

        print('Training done')
        plt.show()
        return print_variable
