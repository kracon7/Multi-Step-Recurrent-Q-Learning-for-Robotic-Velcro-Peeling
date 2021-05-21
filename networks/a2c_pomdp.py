import os
import time
import math
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
import datetime as dt

import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from mujoco_py import MjViewer, load_model_from_path, MjSim
from networks.a2c import ActorCritic
from env import RobotEnv
from utils.memory import RecurrentMemory, Transition
from utils.normalize import Normalizer
from utils.visualization import plot_variables
from utils.gripper_util import init_model


# Class that uses a Deep Recurrent Q-Network to optimize a POMDP
# The assumption here is that the observation o is equal to state s
class A2C_POMDP:
    def __init__(self, args):
        self.args = args
        self.ACTIONS = ['left', 'right', 'forward', 'backward', 'up', 'down']  # 'open', 'close']
        self.max_iter = args.max_iter
        self.gripping_force = args.grip_force
        self.break_threshold = args.break_thresh

        # Prepare the drawing figure
        fig, (ax1, ax2) = plt.subplots(1, 2)
        self.figure = (fig, ax1, ax2)

    # Function to select an action from our policy or a random one
    def select_action(self, obs, hx, cx):

        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            self.model.eval()
            torch_obs = torch.from_numpy(obs).float().to(self.args.device).unsqueeze(0)
            value, logit, (hx, cx) = self.model.forward(torch_obs, hidden_state=hx, cell_state=cx)
            self.model.train()
            prob = F.softmax(logit, dim=-1)
            action = torch.multinomial(prob.view(-1), 1).data

            return value, logit, (hx, cx), prob,  action

    def train_POMDP(self):
        args = self.args
        # Create the output directory if it does not exist
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)

        # Create our policy net and a target net
        self.model = ActorCritic(args.indim, args.outdim).to(args.device) 

        # Set up the optimizer
        self.optimizer = optim.RMSprop(self.model.parameters(), lr=args.lr)
        self.steps_done = 0

        # Setup the state normalizer
        normalizer = Normalizer(num_inputs = args.indim, device=args.device)

        print_variables = {'durations': [], 'rewards': [], 'loss': []}
        start_episode = 0
        if args.checkpoint_file:
            if os.path.exists(args.checkpoint_file):
                checkpoint = torch.load(args.checkpoint_file)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                start_episode = checkpoint['epoch']
                self.steps_done = start_episode
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                with open(os.path.join(os.path.dirname(args.checkpoint_file), 'results_pomdp.pkl'), 'rb') as file:
                    plot_dict = pickle.load(file)
                    print_variables['durations'] = plot_dict['durations']
                    print_variables['rewards'] = plot_dict['rewards']

        if args.normalizer_file:
            if os.path.exists(args.normalizer_file):
                normalizer.restore_state(args.normalizer_file)

        env = RobotEnv(args)
        obs, reward, done, info = env.reset()
        normalizer.observe(obs)
        obs_norm = normalizer.normalize(obs)

        done = True
        episode_length = 0

        # Main training loop
        for ii in range(start_episode, args.epochs):
            start_time = time.time()
            act_sequence =[]

        ########################## Episode Init ################################
            values = []
            log_probs = []
            rewards = []
            entropy_term = 0
            if done:
                hx, cx = self.model.init_hidden_states(args.device)
            else:
                hx = Variable(hx.data, requires_grad=True).to(args.device)
                cx = Variable(hx.data, requires_grad=True).to(args.device)
        ######################################################################

            for step in range(args.max_iter):
                episode_length += 1
                torch_obs = torch.from_numpy(obs_norm).float().to(self.args.device).unsqueeze(0)
                value, policy_dist, (hx, cx) = self.model.forward(torch_obs, hidden_state=hx, cell_state=cx)
                value = value.detach()
                dist = policy_dist.detach().cpu().numpy() 

                action = np.random.choice(len(self.ACTIONS), p=np.squeeze(dist))
                log_prob = torch.log(policy_dist.squeeze(0)[action])
                entropy = -np.sum(np.mean(dist) * np.log(dist))

                new_obs, reward, done, info = env.step(action)
                normalizer.observe(new_obs)
                new_obs_norm = normalizer.normalize(new_obs)

                failure = info['slippage']
                if done:
                    Qval = 0
                if failure or episode_length >= args.max_iter:
                    torch_obs = torch.from_numpy(new_obs_norm).float().to(self.args.device).unsqueeze(0)
                    Qval, _, _ = self.model.forward(torch_obs, hidden_state=hx, cell_state=cx)
                    Qval = Qval.detach().cpu().numpy()[0]

                values.append(value)
                log_probs.append(log_prob)
                rewards.append(reward)
                entropy_term += entropy
                act_sequence.append(action)

                obs_norm = new_obs_norm

                done = done or failure or episode_length >= args.max_iter
                if done:
                    if failure:
                        print_variables['durations'].append(self.max_iter)
                    else:
                        print_variables['durations'].append(episode_length)
                    print_variables['rewards'].append(env.broken_so_far)
                    plot_variables(self.figure, print_variables, "Training POMDP")
                    print("Model parameters: {}".format(env.model_params))
                    print("Actions in this epoch are: {}".format(act_sequence))
                    print("Epoch {} took {}s, total number broken: {}\n\n".format(ii, time.time() - start_time, env.broken_so_far))
                    
                    episode_length = 0
                    obs, _, _, _ = env.reset()
                    normalizer.observe(obs)
                    obs_norm = normalizer.normalize(obs)
                    break

            Qvals = np.zeros_like(values).astype(np.float32)
            for t in reversed(range(len(rewards))):
                Qval = rewards[t] + args.gamma * Qval
                Qvals[t] = Qval

            #update actor critic
            values = torch.FloatTensor(values).to(args.device)
            Qvals = torch.FloatTensor(Qvals).to(args.device)
            log_probs = torch.stack(log_probs).to(args.device)

            advantage = Qvals - values
            actor_loss = (-log_probs * advantage).mean()
            critic_loss = 0.5 * advantage.pow(2).mean()
            ac_loss = actor_loss + critic_loss + 0.01 * entropy_term

            self.optimizer.zero_grad()
            ac_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()
            
            # Save checkpoints every vew iterations
            if ii % args.save_freq == 0:
                save_path = os.path.join(args.output_dir, 'checkpoint_model_' + str(ii) + '.pth')
                torch.save({
                           'epoch': ii,
                           'model_state_dict': self.model.state_dict(),
                           'optimizer_state_dict': self.optimizer.state_dict(),
                           }, save_path)

        # Save normalizer state for inference
        normalizer.save_state(os.path.join(args.output_dir, 'normalizer_state.pickle'))

        if args.savefig_path:
            now = dt.datetime.now()
            self.figure[0].savefig(args.savefig_path+'{}_{}_{}.png'.format(now.month, now.day, now.hour), format='png')

        print('Training done')
        plt.show()
        return print_variables
