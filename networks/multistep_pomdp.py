import os
import time
import math
import random
import pickle
import json
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from itertools import count
import datetime as dt

import torch
import torch.optim as optim
import torch.nn.functional as F
from mujoco_py import MjViewer, load_model_from_path, MjSim
from networks.dqn import DRQN
from networks.tactile_net import TactileNet
from robot_sim import RobotSim
from sim_param import SimParameter
from utils.action_buffer import ActionSpace, TactileObs
from utils.memory import RecurrentMemory, Transition
from utils.normalize import Multimodal_Normalizer
from utils.visualization import plot_variables
from utils.gripper_util import init_model


def write_results(path, results):
    f = open(path, 'wb')
    pickle.dump(results, f)
    f.close()


# Class that uses a Deep Recurrent Q-Network to optimize a POMDP
# The assumption here is that the observation o is equal to state s
class POMDP:
    def __init__(self, args):
        self.args = args
        self.ACTIONS = ['left', 'right', 'forward', 'backward', 'up', 'down']  # 'open', 'close']
        self.P_START = 0.999
        self.P_END = 0.05
        self.P_DECAY = 500
        self.max_iter = args.max_iter
        self.gripping_force = args.grip_force
        self.break_threshold = args.break_thresh

        # Prepare the drawing figure
        fig, (ax1, ax2) = plt.subplots(1, 2)
        self.figure = (fig, ax1, ax2)

    # Function to select an action from our policy or a random one
    def select_action(self, observation, hidden_state, cell_state):
        args = self.args
        sample = random.random()
        p_threshold = self.P_END + (self.P_START - self.P_END) * math.exp(-1. * self.steps_done / self.P_DECAY)

        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            self.tactile_net_1.eval()
            self.policy_net_1.eval()
            torch_obs = torch.from_numpy(observation).float().to(args.device).unsqueeze(0)
            h_tac, c_tac = self.tactile_net_1.init_hidden_states(args.device)
            tactile_ft = self.tactile_net_1.forward(torch_obs, hidden_state=h_tac, cell_state=c_tac)

            model_out = self.policy_net_1(tactile_ft.unsqueeze(1), batch_size=1, time_step=1, hidden_state=hidden_state, cell_state=cell_state)
            out = model_out[0]
            hidden_state = model_out[1][0]
            cell_state = model_out[1][1]
            self.tactile_net_1.train()
            self.policy_net_1.train()

            if sample > p_threshold:
                action = int(torch.argmax(out[0]))
                return action, hidden_state, cell_state
            else:
                return random.randrange(0, args.outdim), hidden_state, cell_state

    def sample_memory(self):
        batch = self.memory.sample(self.args.batch_size, self.args.time_step)
        if not batch:
            return

        current_states, actions, rewards, next_states = [], [], [], []
        for b in batch:
            cs, ac, rw, ns = [], [], [], []
            for element in b:
                cs.append(element.state)
                ac.append(element.action)
                rw.append(element.reward)
                ns.append(element.next_state)
            current_states.append(cs)
            actions.append(ac)
            rewards.append(rw)
            next_states.append(ns)
        return current_states, actions, rewards, next_states

    def extract_ft(self, tactile_net, batch):
        args = self.args
        h_tac, c_tac = tactile_net.init_hidden_states(args.device)
        result = []
        for b in batch:
            obs_sequence = []
            for item in b:
                torch_obs = torch.from_numpy(item).float().to(args.device).unsqueeze(0)
                tactile_ft = tactile_net.forward(torch_obs, hidden_state=h_tac, cell_state=c_tac)
                obs_sequence.append(tactile_ft)
            torch_obs_sequence = torch.stack(obs_sequence)
            result.append(torch_obs_sequence)
        return torch.stack(result)

    def bootstrap(self, policy_net, tactile_net, memory_subset):
        args = self.args
        if len(self.memory) < (args.batch_size):
            return

        current_states, actions, rewards, next_states = memory_subset

        # padded_current_states, current_lengths = self.pad_batch(current_states)
        # padded_next_states, next_lengths = self.pad_batch(next_states)

        # process observation (tactile_obs, img) to 1d tensor of (tactile_obs, feature)
        # and then stack them to corresponding dimension: (batch_size, time_step, *)
        current_features = self.extract_ft(tactile_net, current_states)
        next_features = self.extract_ft(tactile_net, next_states)

        # convert all to torch tensors
        actions = torch.from_numpy(np.array(actions)).long().to(args.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(args.device)

        hidden_batch, cell_batch = policy_net.init_hidden_states(args.batch_size, args.device)

        Q_s, _ = policy_net.forward(current_features,
                                         batch_size=args.batch_size,
                                         time_step=args.time_step,
                                         hidden_state=hidden_batch,
                                         cell_state=cell_batch)
        Q_s_a = Q_s.gather(dim=1, index=actions[:, args.time_step - 1].unsqueeze(dim=1)).squeeze(dim=1)

        Q_next, _ = policy_net.forward(next_features,
                                            batch_size=args.batch_size,
                                            time_step=args.time_step,
                                            hidden_state=hidden_batch,
                                            cell_state=cell_batch)
        Q_next_max = Q_next.detach().max(dim=1)[0]
        return Q_s_a, Q_next_max

    def optimize(self):
        args = self.args
        if len(self.memory) < (args.batch_size):
            return

        memory_subset = self.sample_memory()
        _, _, rewards, _ = memory_subset
        rewards = torch.from_numpy(np.array(rewards)).float().to(args.device)

        Q_s_a_1, Q_next_max_1 = self.bootstrap(self.policy_net_1, self.tactile_net_1, memory_subset)
        Q_s_a_2, Q_next_max_2 = self.bootstrap(self.policy_net_2, self.tactile_net_2, memory_subset)

        Q_next_max = torch.min(Q_next_max_1, Q_next_max_2)

        # Compute the expected Q values
        target_values = rewards[:, args.time_step - 1] + (args.gamma * Q_next_max)

        # Compute Huber loss
        loss_1 = F.smooth_l1_loss(Q_s_a_1, target_values)
        loss_2 = F.smooth_l1_loss(Q_s_a_2, target_values)

        # Optimize the model
        self.policy_optimizer_1.zero_grad()
        self.policy_optimizer_2.zero_grad()
        self.tactile_optimizer_1.zero_grad()
        self.tactile_optimizer_2.zero_grad()
        loss_1.backward()
        loss_2.backward()
        for param in self.policy_net_1.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in self.policy_net_2.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in self.tactile_net_2.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in self.tactile_net_2.parameters():
            param.grad.data.clamp_(-1, 1)
        
        self.policy_optimizer_1.step()
        self.policy_optimizer_2.step()
        self.tactile_optimizer_1.step()
        self.tactile_optimizer_2.step()


    def train_POMDP(self):
        args = self.args
        ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # corl2019
        PARENT_DIR = os.path.dirname(ROOT_DIR)                                  # reserach
        # Create the output directory if it does not exist
        output_dir = os.path.join(PARENT_DIR, 'multistep_pomdp', args.output_dir)
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

        # write args to file
        with open(os.path.join(output_dir, 'args.txt'), 'w+') as f:
            json.dump(args.__dict__, f, indent=2)
        f.close()


        # Create our policy net and a target net
        self.policy_net_1 = DRQN(args.ftdim, args.outdim).to(args.device)        
        self.policy_net_2 = DRQN(args.ftdim, args.outdim).to(args.device)
        if args.position:
            self.tactile_net_1 = TactileNet(args.indim-6, args.ftdim).to(args.device)
            self.tactile_net_2 = TactileNet(args.indim-6, args.ftdim).to(args.device)
        elif args.force:
            self.tactile_net_1 = TactileNet(args.indim-390, args.ftdim).to(args.device)
            self.tactile_net_2 = TactileNet(args.indim-390, args.ftdim).to(args.device)
        else:
            self.tactile_net_1 = TactileNet(args.indim, args.ftdim).to(args.device)
            self.tactile_net_2 = TactileNet(args.indim, args.ftdim).to(args.device)

        # Set up the optimizer
        self.policy_optimizer_1 = optim.RMSprop(self.policy_net_1.parameters(), lr=args.lr)
        self.policy_optimizer_2 = optim.RMSprop(self.policy_net_2.parameters(), lr=args.lr)
        self.tactile_optimizer_1 = optim.RMSprop(self.tactile_net_1.parameters(), lr=args.lr)
        self.tactile_optimizer_2 = optim.RMSprop(self.tactile_net_2.parameters(), lr=args.lr)
        self.memory = RecurrentMemory(800)
        self.steps_done = 0

        # Setup the state normalizer
        normalizer = Multimodal_Normalizer(num_inputs = args.indim, device=args.device)

        print_variables = {'durations': [], 'rewards': [], 'loss': []}
        start_episode = 0
        if args.weight_policy:
            if os.path.exists(args.weight_policy):
                checkpoint = torch.load(args.weight_policy)
                self.policy_net_1.load_state_dict(checkpoint['policy_net_1'])
                self.policy_net_2.load_state_dict(checkpoint['policy_net_2'])
                self.policy_optimizer_1.load_state_dict(checkpoint['policy_optimizer_1'])
                self.policy_optimizer_2.load_state_dict(checkpoint['policy_optimizer_2'])
                start_episode = checkpoint['epochs']
                self.steps_done = checkpoint['steps_done']
                with open(os.path.join(os.path.dirname(args.weight_policy), 'results_pomdp.pkl'), 'rb') as file:
                    plot_dict = pickle.load(file)
                    print_variables['durations'] = plot_dict['durations']
                    print_variables['rewards'] = plot_dict['rewards']

        if args.normalizer_file:
            if os.path.exists(args.normalizer_file):
                normalizer.restore_state(args.normalizer_file)

        if args.memory:
            if os.path.exists(args.memory):
                self.memory.load(args.memory)

        if args.weight_tactile:
            checkpoint = torch.load(args.weight_tactile)
            self.tactile_net_1.load_state_dict(checkpoint['tactile_net_1'])
            self.tactile_optimizer_1.load_state_dict(checkpoint['tactile_optimizer_1'])
            self.tactile_net_2.load_state_dict(checkpoint['tactile_net_2'])
            self.tactile_optimizer_2.load_state_dict(checkpoint['tactile_optimizer_2'])

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

        robot = RobotSim(sim, viewer, sim_param, args.render, self.break_threshold)

        tactile_obs_space = TactileObs(robot.get_gripper_xpos(),            # 24
                             robot.get_all_touch_buffer(args.hap_sample))   # 30 x 6
        
        # Main training loop
        for ii in range(start_episode, args.epochs):
            self.steps_done += 1
            start_time = time.time()
            act_sequence =[]
            act_length = []
            velcro_params = init_model(robot.mj_sim)
            robot.reset_simulation()
            ret = robot.grasp_handle()
            if not ret:
                continue

            # Local memory for current episode
            localMemory = []

            # Get current observation
            hidden_state_1, cell_state_1 = self.policy_net_1.init_hidden_states(batch_size=1, device=args.device)
            hidden_state_2, cell_state_2 = self.policy_net_2.init_hidden_states(batch_size=1, device=args.device)

            broken_so_far = 0

            # pick a random action initially 
            action = random.randrange(0, 5)
            current_state = None
            next_state = None

            t = 0

            while t < args.max_iter:
                if not args.quiet and t == 0:
                    print("Running training episode: {}".format(ii, t))

                if args.position:
                    multistep_obs = np.empty((0, args.indim-6))
                elif args.force:
                    multistep_obs = np.empty((0, args.indim-390))
                else:                      
                    multistep_obs = np.empty((0, args.indim))

                prev_action = action

                for k in range(args.len_ub):
                    # Observe tactile features and stack them
                    tactile_obs = tactile_obs_space.get_state()
                    normalizer.observe(tactile_obs)
                    tactile_obs = normalizer.normalize(tactile_obs)

                    if args.position:
                        tactile_obs = tactile_obs[6:]
                    elif args.force:
                        tactile_obs = tactile_obs[:6]

                    multistep_obs = np.vstack((multistep_obs, tactile_obs))

                    # current jpos
                    current_pos = robot.get_gripper_jpos()[:3]

                    # Perform action
                    delta = action_space.get_action(self.ACTIONS[action])['delta'][:3]
                    target_position = np.add(robot.get_gripper_jpos()[:3], np.array(delta))
                    target_pose = np.hstack((target_position, robot.get_gripper_jpos()[3:]))
                    robot.move_joint(target_pose, True, self.gripping_force, hap_sample = args.hap_sample)

                    # Observe new state
                    tactile_obs_space.update(robot.get_gripper_xpos(),            # 24
                                             robot.get_all_touch_buffer(args.hap_sample))     # 30x6

                    displacement = la.norm(robot.get_gripper_jpos()[:3] - current_pos)

                    if displacement / 0.06 < 0.7:
                        break


                # input stiched multi-step tactile observation into tactile-net to generate tactile feature
                action, hidden_state_1, cell_state_1 = self.select_action(multistep_obs, hidden_state_1, cell_state_1)

                if t == 0:
                    next_state = multistep_obs.copy()
                else:
                    current_state = next_state.copy()
                    next_state = multistep_obs.copy()
                
                # record actions in this epoch
                act_sequence.append(prev_action)
                act_length.append(k)
                
                # Get reward
                done, num = robot.update_tendons()
                failure = robot.check_slippage()
                if num > broken_so_far:
                    reward = num - broken_so_far
                    broken_so_far = num
                else:
                    if failure:
                        reward = -20
                    else:
                        reward = 0
                
                t += k + 1
                # Set max number of iterations
                if t >= self.max_iter:
                    done = True

                if done or failure:
                    next_state = None

                # Push new Transition into memory
                if t > k + 1:
                    localMemory.append(Transition(current_state, prev_action, next_state, reward))

                # Optimize the model
                if self.steps_done % 10 == 0:
                    self.optimize()

                # If we are done, reset the model
                if done or failure:
                    self.memory.push(localMemory)
                    if failure:
                        print_variables['durations'].append(self.max_iter)
                    else:
                        print_variables['durations'].append(t)
                    print_variables['rewards'].append(broken_so_far)
                    plot_variables(self.figure, print_variables, "Training POMDP")
                    print("Model parameters: {}".format(velcro_params))
                    print("{} of Actions in this epoch are: {} \n Action length are: {}".format(len(act_sequence), act_sequence, act_length))
                    print("Epoch {} took {}s, total number broken: {}\n\n".format(ii, time.time() - start_time, broken_so_far))

                    break

            # Save checkpoints every vew iterations
            if ii % args.save_freq == 0:
                save_path = os.path.join(output_dir, 'policy_' + str(ii) + '.pth')
                torch.save({
                           'epochs': ii,
                           'steps_done': self.steps_done,
                           'policy_net_1': self.policy_net_1.state_dict(),
                           'policy_net_2': self.policy_net_2.state_dict(),
                           'policy_optimizer_1': self.policy_optimizer_1.state_dict(),
                           'policy_optimizer_2': self.policy_optimizer_2.state_dict(),
                           }, save_path)
                save_path = os.path.join(output_dir, 'tactile_' + str(ii) + '.pth')
                torch.save({
                           'tactile_net_1': self.tactile_net_1.state_dict(),
                           'tactile_net_2': self.tactile_net_2.state_dict(),
                           'tactile_optimizer_1': self.tactile_optimizer_1.state_dict(),
                           'tactile_optimizer_2': self.tactile_optimizer_2.state_dict(),
                           }, save_path)

                write_results(os.path.join(output_dir, 'results_pomdp.pkl'), print_variables)

                self.memory.save_memory(os.path.join(output_dir, 'memory.pickle'))

        if args.savefig_path:
            now = dt.datetime.now()
            self.figure[0].savefig(args.savefig_path+'{}_{}_{}.png'.format(now.month, now.day, now.hour), format='png')

        print('Training done')
        plt.show()
        return print_variables
