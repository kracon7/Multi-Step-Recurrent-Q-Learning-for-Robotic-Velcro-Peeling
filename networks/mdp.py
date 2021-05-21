import os
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
import datetime as dt

import torch
import torch.optim as optim
import torch.nn.functional as F
from mujoco_py import MjViewer, MjSim
from networks.dqn import DQN
from robot_sim import RobotSim
from sim_param import SimParameter
from utils.action_buffer import ActionSpace, Observation
from utils.memory import ReplayMemory, Transition
from utils.normalize import Normalizer
from utils.visualization import plot_variables
from utils.gripper_util import init_model


# Class that uses a Deep Q-Network to optimize a MDP
# The assumption here is that the observation o is equal to state s
class MDP:
    def __init__(self, args):
        self.args = args
        self.ACTIONS = ['left', 'right', 'forward', 'backward', 'up', 'down']  # 'open', 'close']
        self.P_START = 0.999
        self.P_END = 0.05
        self.P_DECAY = 500
        self.max_iter = args.max_iter
        self.gripping_force = args.grip_force
        self.breaking_threshold = args.break_thresh

        # Prepare the drawing figure
        fig, (ax1, ax2) = plt.subplots(1, 2)
        self.figure = (fig, ax1, ax2)

    # Function to select an action from our policy or a random one
    def select_action(self, state):
        sample = random.random()
        p_threshold = self.P_END + (self.P_START - self.P_END) * math.exp(-1. * self.steps_done / self.P_DECAY)
        self.steps_done += 1

        if sample > p_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                self.policy_net_1.eval()
                torch_state = torch.from_numpy(state).float().to(self.args.device)
                action = self.policy_net_1(torch_state.unsqueeze(0)).max(1)[1]
                self.policy_net_1.train()
                return action.item()
        else:
            return random.randrange(self.args.outdim)

    def optimize_model(self):
        args = self.args
        if len(self.memory) < args.batch_size:
            return

        transitions = self.memory.sample(args.batch_size)

        state_batch, action_batch, reward_batch, nextstate_batch = [], [], [], []
        for transition in transitions:
            state_batch.append(transition.state)
            action_batch.append(transition.action)
            reward_batch.append(transition.reward)
            nextstate_batch.append(transition.next_state)

        state_batch = torch.from_numpy(np.array(state_batch)).float().to(args.device)
        action_batch = torch.from_numpy(np.array(action_batch)).to(args.device).unsqueeze(1)
        reward_batch = torch.from_numpy(np.array(reward_batch)).float().to(args.device).unsqueeze(1)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, nextstate_batch)),
                                      device=args.device, dtype=torch.bool).unsqueeze(1)
        non_final_next_states = torch.cat([torch.from_numpy(s).float().to(args.device).unsqueeze(0) for s in nextstate_batch if s is not None])

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values_1 = self.policy_net_1(state_batch).gather(1, action_batch)
        state_action_values_2 = self.policy_net_2(state_batch).gather(1, action_batch)
        state_action_values_3 = self.policy_net_3(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values_1 = torch.zeros((args.batch_size, 1), device=args.device)
        next_state_values_2 = torch.zeros((args.batch_size, 1), device=args.device)
        next_state_values_3 = torch.zeros((args.batch_size, 1), device=args.device)
        next_state_values_1[non_final_mask] = self.policy_net_1(non_final_next_states).max(1)[0].detach()
        next_state_values_2[non_final_mask] = self.policy_net_2(non_final_next_states).max(1)[0].detach()
        next_state_values_3[non_final_mask] = self.policy_net_3(non_final_next_states).max(1)[0].detach()

        next_state_values = torch.min(torch.min(next_state_values_1, next_state_values_2), next_state_values_3)

        # Compute the expected Q values
        expected_state_action_values = (next_state_values * args.gamma) + reward_batch

        # Compute Huber loss
        loss_1 = F.smooth_l1_loss(state_action_values_1, expected_state_action_values)
        loss_2 = F.smooth_l1_loss(state_action_values_2, expected_state_action_values)
        loss_3 = F.smooth_l1_loss(state_action_values_3, expected_state_action_values)

        # Optimize the model
        self.optimizer_1.zero_grad()
        self.optimizer_2.zero_grad()
        self.optimizer_3.zero_grad()
        loss_1.backward()
        loss_2.backward()
        loss_3.backward()
        for param in self.policy_net_1.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in self.policy_net_2.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in self.policy_net_3.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer_1.step()
        self.optimizer_2.step()
        self.optimizer_3.step()
        return [loss_1, loss_2, loss_3]

    def train_MDP(self):
        args = self.args
        # Create the output directory if it does not exist
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)

        # Create our policy net and a target net
        self.policy_net_1 = DQN(args.indim, args.outdim).to(args.device)
        self.policy_net_2 = DQN(args.indim, args.outdim).to(args.device)
        self.policy_net_3 = DQN(args.indim, args.outdim).to(args.device)
        
        self.target_net = DQN(args.indim, args.outdim).to(args.device)
        self.target_net.load_state_dict(self.policy_net_1.state_dict())
        self.target_net.eval()

        # Set up the optimizer
        self.optimizer_1 = optim.RMSprop(self.policy_net_1.parameters(), args.lr)
        self.optimizer_2 = optim.RMSprop(self.policy_net_2.parameters(), args.lr)
        self.optimizer_3 = optim.RMSprop(self.policy_net_3.parameters(), args.lr)
        self.memory = ReplayMemory(500000)
        self.steps_done = 0

        # Setup the state normalizer
        normalizer = Normalizer(args.indim, device=args.device)
        print_variables = {'durations': [], 'rewards': [], 'loss': []}

        # Load old checkpoint if provided
        start_episode = 0
        if args.checkpoint_file:
            if os.path.exists(args.checkpoint_file):
                checkpoint = torch.load(args.checkpoint_file)
                self.policy_net_1.load_state_dict(checkpoint['model_state_dict'])
                self.policy_net_2.load_state_dict(checkpoint['model_state_dict'])
                self.policy_net_3.load_state_dict(checkpoint['model_state_dict'])
                self.target_net.load_state_dict(checkpoint['model_state_dict'])
                start_episode = checkpoint['epoch']
                self.steps_done = start_episode
                self.optimizer_1.load_state_dict(checkpoint['optimizer_state_dict'])
                self.optimizer_2.load_state_dict(checkpoint['optimizer_state_dict'])
                self.optimizer_3.load_state_dict(checkpoint['optimizer_state_dict'])
                with open(os.path.join(os.path.dirname(args.checkpoint_file), 'results_geom_mdp.pkl'), 'rb') as file:
                    plot_dict = pickle.load(file)
                    print_variables['durations'] = plot_dict['durations']
                    print_variables['rewards'] = plot_dict['rewards']

        if args.normalizer_file:
            if os.path.exists(args.normalizer_file):
                normalizer.restore_state(args.normalizer_file)

        action_space = ActionSpace(dp=0.06, df=10)

        # Main training loop
        for ii in range(start_episode, args.epochs):
            start_time = time.time()
            if args.sim:
                # Create robot, reset simulation and grasp handle
                model, model_params = init_model(args.model_path)
                sim = MjSim(model)
                sim.step()
                viewer = None
                if args.render:
                    viewer = MjViewer(sim)
                else:
                    viewer = None

                sim_param = SimParameter(sim)
                robot = RobotSim(sim, viewer, sim_param, args.render, self.breaking_threshold)
                robot.reset_simulation()
                ret = robot.grasp_handle()
                if not ret:
                    continue

                # Get current state
                state_space = Observation(robot.get_gripper_jpos(),
                                          robot.get_shear_buffer(args.hap_sample),
                                          robot.get_all_touch_buffer(args.hap_sample))
                broken_so_far = 0

            for t in count():
                if not args.quiet and t % 20 == 0:
                    print("Running training episode: {}, iteration: {}".format(ii, t))

                # Select action
                state = state_space.get_state()
                if args.position:
                    state = state[6:]
                if args.shear:
                    indices = np.ones(len(state), dtype=bool)
                    indices[6:166] = False
                    state = state[indices]
                if args.force:
                    state = state[:166]
                normalizer.observe(state)
                state = normalizer.normalize(state)
                action = self.select_action(state)

                # Perform action
                delta = action_space.get_action(self.ACTIONS[action])['delta'][:3]
                target_position = np.add(state_space.get_current_position(), np.array(delta))
                target_pose = np.hstack((target_position, robot.get_gripper_jpos()[3:]))

                if args.sim:
                    robot.move_joint(target_pose, True, self.gripping_force, hap_sample = args.hap_sample)

                    # Get reward
                    done, num = robot.update_tendons()
                    failure = robot.check_slippage()
                    if num > broken_so_far:
                        reward = num - broken_so_far
                        broken_so_far = num
                    else:
                        reward = 0

                    # # Add a movement reward
                    # reward -= 0.1 * np.linalg.norm(target_position - robot.get_gripper_jpos()[:3]) / np.linalg.norm(delta)

                    # Observe new state
                    state_space.update(robot.get_gripper_jpos(),
                                       robot.get_shear_buffer(args.hap_sample),
                                       robot.get_all_touch_buffer(args.hap_sample))

                # Set max number of iterations
                if t >= self.max_iter:
                    done = True

                # Check if done
                if not done and not failure:
                    next_state = state_space.get_state()
                    if args.position:
                        next_state = next_state[6:]
                    if args.shear:
                        indices = np.ones(len(next_state), dtype=bool)
                        indices[6:166] = False
                        next_state = next_state[indices]
                    if args.force:
                        next_state = next_state[:166]
                    normalizer.observe(next_state)
                    next_state = normalizer.normalize(next_state)
                else:
                    next_state = None

                # Push new Transition into memory
                self.memory.push(state, action, next_state, reward)

                # Optimize the model
                loss = self.optimize_model()
        #        if loss:
        #            print_variables['loss'].append(loss.item())

                # If we are done, reset the model
                if done or failure:
                    if failure:
                        print_variables['durations'].append(self.max_iter)
                    else:
                        print_variables['durations'].append(t)
                    print_variables['rewards'].append(broken_so_far)
                    plot_variables(self.figure, print_variables, 'Training MDP')
                    print("Model parameters: {}".format(model_params))
                    print("Epoch {} took {}s, total number broken: {}\n\n".format(ii, time.time() - start_time, broken_so_far))
                    break

            # Update the target network, every x iterations
            if ii % 10 == 0:
                self.target_net.load_state_dict(self.policy_net_1.state_dict())

            # Save checkpoints every vew iterations
            if ii % args.save_freq == 0:
                save_path = os.path.join(args.output_dir, 'checkpoint_model_' + str(ii) + '.pth')
                torch.save({
                           'epoch': ii,
                           'model_state_dict': self.target_net.state_dict(),
                           'optimizer_state_dict': self.optimizer_1.state_dict(),
                           }, save_path)

        # Save normalizer state for inference
        normalizer.save_state(os.path.join(args.output_dir, 'normalizer_state.pickle'))

        if args.savefig_path:
            now = dt.datetime.now()
            self.figure[0].savefig(args.savefig_path+'{}_{}_{}'.format(now.month, now.day, now.hour), format='png')

        print('Training done')
        plt.show()
        return print_variables
