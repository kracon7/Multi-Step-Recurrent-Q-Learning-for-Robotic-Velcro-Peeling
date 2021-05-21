import os
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
import pickle
import datetime as dt

import torch
import torch.optim as optim
import torch.nn.functional as F
from mujoco_py import MjViewer, MjSim, load_model_from_path
from networks.dqn import Geom_DQN
from robot_sim import RobotSim
from sim_param import SimParameter
from utils.action_buffer import ActionSpace, Observation
from utils.memory import ReplayMemory, Transition
from utils.normalize import Geom_Normalizer
from utils.visualization import plot_variables
from utils.gripper_util import init_model, change_sim
from utils.velcro_utils import VelcroUtil


# Class that uses a Deep Q-Network to optimize a MDP
# The assumption here is that the observation o is equal to state s
class Geom_MDP:
    def __init__(self, args):
        self.args = args
        self.ACTIONS = ['left', 'right', 'forward', 'backward', 'up', 'down']  # 'open', 'close']
        self.P_START = 0.999
        self.P_END = 0.05
        self.P_DECAY = 50000
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

        # Create two policy net and one target net
        self.policy_net_1 = Geom_DQN(args.indim, args.outdim).to(args.device)
        self.policy_net_2 = Geom_DQN(args.indim, args.outdim).to(args.device)
        self.policy_net_3 = Geom_DQN(args.indim, args.outdim).to(args.device)

        # Set up the optimizer
        self.optimizer_1 = optim.RMSprop(self.policy_net_1.parameters(), args.lr)
        self.optimizer_2 = optim.RMSprop(self.policy_net_2.parameters(), args.lr)
        self.optimizer_3 = optim.RMSprop(self.policy_net_3.parameters(), args.lr)
        self.memory = ReplayMemory(500000)
        self.steps_done = 0

        # Setup the state normalizer
        normalizer = Geom_Normalizer(args.indim - 12, device=args.device)
        print_variables = {'durations': [], 'rewards': []}

        # Load old checkpoint if provided
        start_episode = 0
        if args.checkpoint_file:
            if os.path.exists(args.checkpoint_file):
                checkpoint = torch.load(args.checkpoint_file)
                self.policy_net_1.load_state_dict(checkpoint['policy_net_1'])
                self.policy_net_2.load_state_dict(checkpoint['policy_net_2'])
                self.policy_net_3.load_state_dict(checkpoint['policy_net_3'])
                start_episode = checkpoint['epochs']
                self.steps_done = checkpoint['steps_done']
                self.optimizer_1.load_state_dict(checkpoint['optimizer_1'])
                self.optimizer_2.load_state_dict(checkpoint['optimizer_2'])
                self.optimizer_3.load_state_dict(checkpoint['optimizer_3'])
                with open(os.path.join(os.path.dirname(args.checkpoint_file), 'results_geom_mdp.pkl'), 'rb') as file:
                    plot_dict = pickle.load(file)
                    print_variables['durations'] = plot_dict['durations']
                    print_variables['rewards'] = plot_dict['rewards']

        if args.normalizer_file:
        	if os.path.exists(args.normalizer_file):
        		normalizer.restore_state(args.normalizer_file)

        if args.memory:
            if os.path.exists(args.memory):
                self.memory.load(args.memory)
       
        action_space = ActionSpace(dp=0.06, df=10)

        # Create robot, reset simulation and grasp handle
        model = load_model_from_path(args.model_path)
        sim = MjSim(model)
        sim_param = SimParameter(sim)
        sim.step()
        if args.render:
            viewer = MjViewer(sim)
        else:
            viewer = None
        robot = RobotSim(sim, viewer, sim_param, args.render, args.break_thresh)
        # load all velcro parameters
        model_dir = os.path.dirname(args.model_path)
        param_path = os.path.join(model_dir, 'uniform_sample.pkl')
        velcro_params = pickle.load(open(param_path, 'rb'))

        velcro_util = VelcroUtil(robot, sim_param)
        state_space = Observation(robot.get_gripper_jpos(),  # 6
                                          velcro_util.break_center(),         # 6
                                          velcro_util.break_norm())

        # Main training loop
        for ii in range(start_episode, args.epochs):
            start_time = time.time()
            act_sequence =[]

            geom_type, origin_offset, euler, radius = random.sample(velcro_params, 1)[0]
            change_sim(robot.mj_sim, geom_type, origin_offset, euler, radius)

            robot.reset_simulation()
            ret = robot.grasp_handle()
            if not ret:
                continue

            # Get current state
            state_space.update(robot.get_gripper_jpos(),  # 6
                                          velcro_util.break_center(),         # 6
                                          velcro_util.break_norm())  # 12
            broken_so_far = 0

            for t in count():
                if not args.quiet and t % 20 == 0:
                    print("Running training episode: {}, iteration: {}".format(ii, t))

                # Select action
                state = state_space.get_state()
                normalizer.observe(state[:12])
                state[:12] = normalizer.normalize(state[:12])
                action = self.select_action(state)
                act_sequence.append(action)

                # Perform action
                delta = action_space.get_action(self.ACTIONS[action])['delta'][:3]
                target_position = np.add(robot.get_gripper_jpos()[:3], np.array(delta))
                target_pose = np.hstack((target_position, robot.get_gripper_jpos()[3:]))

                robot.move_joint(target_pose, True, self.gripping_force, hap_sample = args.hap_sample)

                # Get reward
                done, num = robot.update_tendons()
                failure = robot.check_slippage()
                if num > broken_so_far:
                    reward = num - broken_so_far
                    broken_so_far = num
                else:
                    reward = 0

                # Observe new state
                state_space.update(robot.get_gripper_jpos(),  # 6
                                          velcro_util.break_center(),         # 6
                                          velcro_util.break_norm())  # 12

                # Set max number of iterations
                if t >= self.max_iter:
                    done = True

                # Check if done
                if not done and not failure:
                    next_state = state_space.get_state()
                    normalizer.observe(next_state[:12])
                    next_state[:12] = normalizer.normalize(next_state[:12])
                else:
                    next_state = None

                # Push new Transition into memory
                self.memory.push(state, action, next_state, reward)

                # Optimize the model  
                if self.steps_done % 10 == 0:
                    loss = self.optimize_model()

                # If we are done, reset the model
                if done or failure:
                    if failure:
                        print_variables['durations'].append(self.max_iter)
                    else:
                        print_variables['durations'].append(t)
                    print_variables['rewards'].append(broken_so_far)
                    plot_variables(self.figure, print_variables, 'Training MDP')
                    print("Model parameters: {} {} {} {}".format(geom_type, origin_offset, euler, radius))
                    print("Actions in this epoch are: {}".format(act_sequence))
                    print("Epoch {} took {}s, total number broken: {}\n\n".format(ii, time.time() - start_time, broken_so_far))
                    break

            # Save checkpoints every vew iterations
            if ii % args.save_freq == 0:
                save_path = os.path.join(args.output_dir, 'checkpoint_model_' + str(ii) + '.pth')
                torch.save({
                            'epochs': ii,
                            'steps_done': self.steps_done,
                            'policy_net_1': self.policy_net_1.state_dict(),
                            'policy_net_2': self.policy_net_2.state_dict(),
                            'policy_net_3': self.policy_net_3.state_dict(),
                            'optimizer_1': self.optimizer_1.state_dict(),
                            'optimizer_2': self.optimizer_2.state_dict(),
                            'optimizer_3': self.optimizer_3.state_dict(),
                           }, save_path)

        # Save normalizer state for inference
        normalizer.save_state(os.path.join(args.output_dir, 'normalizer_state.pickle'))
        self.memory.save_memory(os.path.join(args.output_dir, 'memory.pickle'))

        if args.savefig_path:
            now = dt.datetime.now()
            self.figure[0].savefig(args.savefig_path+'{}_{}_{}'.format(now.month, now.day, now.hour), format='png')

        print('Training done')
        plt.show()
        return print_variables