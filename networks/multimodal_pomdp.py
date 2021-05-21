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
from mujoco_py import MjViewer, load_model_from_path, MjSim
from networks.dqn import DRQN
from networks.conv_net import ConvNet
from robot_sim import RobotSim
from sim_param import SimParameter
from utils.action_buffer import ActionSpace, TactileObs
from utils.memory import RecurrentMemory, Transition
from utils.normalize import Multimodal_Normalizer
from utils.visualization import plot_variables
from utils.gripper_util import init_model, norm_img, norm_depth


# Class that uses a Deep Recurrent Q-Network to optimize a POMDP
# The assumption here is that the observation o is equal to state s
class POMDP:
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
        fig, (ax1, ax2) = plt.subplots(1, 2)
        self.figure = (fig, ax1, ax2)

    # Function to select an action from our policy or a random one
    def select_action(self, observation, hidden_state, cell_state):
        args = self.args
        sample = random.random()
        p_threshold = self.P_END + (self.P_START - self.P_END) * math.exp(-1. * self.steps_done / self.P_DECAY)
        self.steps_done += 1

        tactile_obs, img_norm = observation

        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            self.policy_net_1.eval()
            torch_tactile_obs = torch.from_numpy(tactile_obs).float().to(args.device).unsqueeze(0)
            torch_img_norm = torch.from_numpy(img_norm).float().to(args.device).unsqueeze(0)
            img_ft = self.conv_net_1.forward(torch_img_norm)

            torch_observation = torch.cat((torch_tactile_obs, img_ft), dim=1)
            model_out = self.policy_net_1(torch_observation, batch_size=1, time_step=1, hidden_state=hidden_state, cell_state=cell_state)
            out = model_out[0]
            hidden_state = model_out[1][0]
            cell_state = model_out[1][1]
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
            result.append(obs_squence)
        return torch.stack(result)

    def bootstrap(self, policy_net, conv_net, memory_subset):
        args = self.args
        if len(self.memory) < (args.batch_size):
            return

        current_states, actions, rewards, next_states = memory_subset

        # process observation (tactile_obs, img) to 1d tensor of (tactile_obs, feature)
        # and then stack them to corresponding dimension: (batch_size, time_step, *)
        current_states = self.extract_ft(conv_net, current_states)
        next_states = self.extract_ft(conv_net, next_states)

        # convert all to torch tensors
        actions = torch.from_numpy(np.array(actions)).long().to(args.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(args.device)

        hidden_batch, cell_batch = policy_net.init_hidden_states(args.batch_size, args.device)

        Q_s, _ = policy_net.forward(current_states,
                                         batch_size=args.batch_size,
                                         time_step=args.time_step,
                                         hidden_state=hidden_batch,
                                         cell_state=cell_batch)
        Q_s_a = Q_s.gather(dim=1, index=actions[:, args.time_step - 1].unsqueeze(dim=1)).squeeze(dim=1)

        Q_next, _ = policy_net.forward(next_states,
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

        Q_s_a_1, Q_next_max_1 = self.bootstrap(self.policy_net_1, self.conv_net_1, memory_subset)
        Q_s_a_2, Q_next_max_2 = self.bootstrap(self.policy_net_2, self.conv_net_2, memory_subset)

        Q_next_max = torch.min(Q_next_max_1, Q_next_max_2)

        # Compute the expected Q values
        target_values = rewards[:, args.time_step - 1] + (args.gamma * Q_next_max)

        # Compute Huber loss
        loss_1 = F.smooth_l1_loss(Q_s_a_1, target_values)
        loss_2 = F.smooth_l1_loss(Q_s_a_2, target_values)

        # Optimize the model
        self.policy_optimizer_1.zero_grad()
        self.policy_optimizer_2.zero_grad()
        self.conv_optimizer_1.zero_grad()
        self.conv_optimizer_2.zero_grad()
        loss_1.backward()
        loss_2.backward()
        for param in self.policy_net_1.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in self.policy_net_2.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in self.conv_net_2.parameters():
            param.grad.data.clamp_(-1, 1)
        for param in self.conv_net_2.parameters():
            param.grad.data.clamp_(-1, 1)
        
        self.policy_optimizer_1.step()
        self.policy_optimizer_2.step()
        self.conv_optimizer_1.step()
        self.conv_optimizer_2.step()


    def train_POMDP(self):
        args = self.args
        # Create the output directory if it does not exist
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)

        # Create our policy net and a target net
        self.policy_net_1 = DRQN(args.indim, args.outdim).to(args.device)        
        self.policy_net_2 = DRQN(args.indim, args.outdim).to(args.device)
        self.conv_net_1 = ConvNet(args.ftdim, args.depth).to(args.device)
        self.conv_net_2 = ConvNet(args.ftdim, args.depth).to(args.device)

        # Set up the optimizer
        self.policy_optimizer_1 = optim.RMSprop(self.policy_net_1.parameters(), lr=args.lr)
        self.policy_optimizer_2 = optim.RMSprop(self.policy_net_2.parameters(), lr=args.lr)
        self.conv_optimizer_1 = optim.RMSprop(self.conv_net_1.parameters(), lr=1e-5)
        self.conv_optimizer_2 = optim.RMSprop(self.conv_net_2.parameters(), lr=1e-5)
        self.memory = RecurrentMemory(70)
        self.steps_done = 0

        # Setup the state normalizer
        normalizer = Multimodal_Normalizer(num_inputs = args.indim - args.ftdim, device=args.device)

        print_variables = {'durations': [], 'rewards': [], 'loss': []}
        start_episode = 0
        if args.checkpoint_file:
            if os.path.exists(args.checkpoint_file):
                checkpoint = torch.load(args.checkpoint_file)
                self.policy_net_1.load_state_dict(checkpoint['policy_net_1'])
                self.policy_net_2.load_state_dict(checkpoint['policy_net_2'])
                self.conv_net_1.load_state_dict(checkpoint['conv_net_1'])
                self.conv_net_2.load_state_dict(checkpoint['conv_net_2'])
                self.policy_optimizer_1.load_state_dict(checkpoint['policy_optimizer_1'])
                self.policy_optimizer_2.load_state_dict(checkpoint['policy_optimizer_2'])
                self.conv_optimizer_1.load_state_dict(checkpoint['conv_optimizer_1'])
                self.conv_optimizer_2.load_state_dict(checkpoint['conv_optimizer_2'])
                start_episode = checkpoint['epoch']
                self.steps_done = checkpoint['steps_done']
                with open(os.path.join(os.path.dirname(args.checkpoint_file), 'results_pomdp.pkl'), 'rb') as file:
                    plot_dict = pickle.load(file)
                    print_variables['durations'] = plot_dict['durations']
                    print_variables['rewards'] = plot_dict['rewards']

        if args.normalizer_file:
            if os.path.exists(args.normalizer_file):
                normalizer.restore_state(args.normalizer_file)

        if args.memory:
            if os.path.exists(args.memory):
                self.memory.load(args.memory)

        if args.weight_conv:
            checkpoint = torch.load(args.weight_conv)
            self.conv_net_1.load_state_dict(checkpoint['conv_net'])
            self.conv_optimizer_1.load_state_dict(checkpoint['conv_optimizer'])
            self.conv_net_2.load_state_dict(checkpoint['conv_net'])
            self.conv_optimizer_2.load_state_dict(checkpoint['conv_optimizer'])

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
        tactile_obs_space = TactileObs(robot.get_gripper_jpos(),            # 6
                             robot.get_all_touch_buffer(args.hap_sample))   # 30 x 12
        
        # Main training loop
        for ii in range(start_episode, args.epochs):
            start_time = time.time()
            act_sequence =[]
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

            for t in count():
                if not args.quiet and t % 50 == 0:
                    print("Running training episode: {}, iteration: {}".format(ii, t))

                # Select action
                tactile_obs = tactile_obs_space.get_state()
                normalizer.observe(tactile_obs)
                tactile_obs = normalizer.normalize(tactile_obs)
                # Get image and normalize it
                img = robot.get_img(args.img_w, args.img_h, 'c1', args.depth)
                if args.depth:
                    depth = norm_depth(img[1])
                    img = norm_img(img[0])
                    img_norm = np.empty((4, args.img_w, args.img_h))
                    img_norm[:3,:,:] = img
                    img_norm[3,:,:] = depth
                else:
                    img_norm = norm_img(img)

                observation = [tactile_obs, img_norm]
                action, hidden_state_1, cell_state_1 = self.select_action(observation, hidden_state_1, cell_state_1)
                
                # record actions in this epoch
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
                tactile_obs_space.update(robot.get_gripper_jpos(),            # 6
                                         robot.get_all_touch_buffer(args.hap_sample))     # 30x12

                # Set max number of iterations
                if t >= self.max_iter:
                    done = True

                # Check if done
                if not done and not failure:
                    next_tactile_obs = tactile_obs_space.get_state()
                    normalizer.observe(next_tactile_obs)
                    next_tactile_obs = normalizer.normalize(next_tactile_obs)
                    # Get image and normalize it
                    next_img = robot.get_img(args.img_w, args.img_h, 'c1', args.depth)
                    if args.depth:
                        next_depth = norm_depth(next_img[1])
                        next_img = norm_img(next_img[0])
                        next_img_norm = np.empty((4, args.img_w, args.img_h))
                        next_img_norm[:3,:,:] = next_img
                        next_img_norm[3,:,:] = next_depth
                    else:
                        next_img_norm = norm_img(next_img)
                    next_state = [next_tactile_obs, next_img_norm]
                else:
                    next_state = None

                # Push new Transition into memory
                localMemory.append(Transition(observation, action, next_state, reward))

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
                    print("Actions in this epoch are: {}".format(act_sequence))
                    print("Epoch {} took {}s, total number broken: {}\n\n".format(ii, time.time() - start_time, broken_so_far))

                    break

            # Save checkpoints every vew iterations
            if ii % args.save_freq == 0:
                save_path = os.path.join(args.output_dir, 'checkpoint_model_' + str(ii) + '.pth')
                torch.save({
                           'epochs': ii,
                           'steps_done': self.steps_done,
                           'conv_net_1': self.conv_net_1.state_dict(),
                           'conv_net_2': self.conv_net_2.state_dict(),
                           'policy_net_1': self.policy_net_1.state_dict(),
                           'policy_net_2': self.policy_net_2.state_dict(),
                           'conv_optimizer_1': self.conv_optimizer_1.state_dict(),
                           'conv_optimizer_2': self.conv_optimizer_2.state_dict(),
                           'policy_optimizer_1': self.policy_optimizer_1.state_dict(),
                           'policy_optimizer_2': self.policy_optimizer_2.state_dict(),
                           }, save_path)

        # Save normalizer state for inference
        normalizer.save_state(os.path.join(args.output_dir, 'normalizer_state.pickle'))

        self.memory.save_memory(os.path.join(args.output_dir, 'memory.pickle'))

        if args.savefig_path:
            now = dt.datetime.now()
            self.figure[0].savefig(args.savefig_path+'{}_{}_{}.png'.format(now.month, now.day, now.hour), format='png')

        print('Training done')
        plt.show()
        return print_variables
