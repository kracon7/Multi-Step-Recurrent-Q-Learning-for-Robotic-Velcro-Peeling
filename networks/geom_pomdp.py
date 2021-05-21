import os
import time
import math
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from itertools import count

import torch
import torch.optim as optim
import torch.nn.functional as F
from mujoco_py import MjViewer, load_model_from_path, MjSim
from networks.dqn import Geom_DRQN
from robot_sim import RobotSim
from sim_param import SimParameter
from utils.action_buffer import ActionSpace, Observation
from utils.memory import RecurrentMemory, Transition
from utils.normalize import Geom_Normalizer
from utils.visualization import plot_variables
from utils.gripper_util import init_model
from utils.velcro_utils import VelcroUtil


# Class that uses a Deep Recurrent Q-Network to optimize a POMDP
# The assumption here is that the observation o is equal to state s
class Geom_POMDP:
    def __init__(self, args):
        self.args = args
        self.ACTIONS = ['left', 'right', 'forward', 'backward', 'up', 'down']  # 'open', 'close']
        self.P_START = 0.999
        self.P_END = 0.05
        self.P_DECAY = 400
        self.max_iter = args.max_iter
        self.gripping_force = args.grip_force
        self.break_threshold = args.break_thresh

        # Prepare the drawing figure
        fig, (ax1, ax2) = plt.subplots(1, 2)
        self.figure = (fig, ax1, ax2)

    # Function to select an action from our policy or a random one
    def select_action(self, observation, hidden_state, cell_state):
        sample = random.random()
        p_threshold = self.P_END + (self.P_START - self.P_END) * math.exp(-1. * self.steps_done / self.P_DECAY)
        self.steps_done += 1

        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            self.policy_net.eval()
            torch_observation = torch.from_numpy(observation).float().to(self.args.device).unsqueeze(0)
            model_out = self.policy_net(torch_observation, batch_size=1, time_step=1, hidden_state=hidden_state, cell_state=cell_state)
            out = model_out[0]
            hidden_state = model_out[1][0]
            cell_state = model_out[1][1]
            self.policy_net.train()

            if sample > p_threshold:
                action = int(torch.argmax(out[0]))
                return action, hidden_state, cell_state
            else:
                return random.randrange(0, self.args.outdim), hidden_state, cell_state

    def optimize_model(self):
        args = self.args
        if len(self.memory) < (args.batch_size):
            return

        hidden_batch, cell_batch = self.policy_net.init_hidden_states(args.batch_size, args.device)
        batch = self.memory.sample(args.batch_size, args.time_step)
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

        current_states = torch.from_numpy(np.array(current_states)).float().to(args.device)
        actions = torch.from_numpy(np.array(actions)).long().to(args.device)
        rewards = torch.from_numpy(np.array(rewards)).float().to(args.device)
        next_states = torch.from_numpy(np.array(next_states)).float().to(args.device)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        Q_s, _ = self.policy_net.forward(current_states,
                                         batch_size=args.batch_size,
                                         time_step=args.time_step,
                                         hidden_state=hidden_batch,
                                         cell_state=cell_batch)
        Q_s_a = Q_s.gather(dim=1, index=actions[:, args.time_step - 1].unsqueeze(dim=1)).squeeze(dim=1)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        Q_next, _ = self.target_net.forward(next_states,
                                            batch_size=args.batch_size,
                                            time_step=args.time_step,
                                            hidden_state=hidden_batch,
                                            cell_state=cell_batch)
        Q_next_max = Q_next.detach().max(dim=1)[0]

        # Compute the expected Q values
        target_values = rewards[:, args.time_step - 1] + (args.gamma * Q_next_max)

        # Compute Huber loss
        loss = F.smooth_l1_loss(Q_s_a, target_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        return loss

    def train_POMDP(self):
        args = self.args
        # Create the output directory if it does not exist
        if not os.path.isdir(args.output_dir):
            os.makedirs(args.output_dir)

        # Create our policy net and a target net
        self.policy_net = Geom_DRQN(args.indim, args.outdim).to(args.device)
        self.target_net = Geom_DRQN(args.indim, args.outdim).to(args.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Set up the optimizer
        self.optimizer = optim.RMSprop(self.policy_net.parameters())
        self.memory = RecurrentMemory(2000)
        self.steps_done = 0

        # Setup the state normalizer
        normalizer = Geom_Normalizer(args.indim - 12, device=args.device)

        print_variables = {'durations': [], 'rewards': [], 'loss': []}
        start_episode = 0
        if args.checkpoint_file:
            if os.path.exists(args.checkpoint_file):
                checkpoint = torch.load(args.checkpoint_file)
                self.policy_net.load_state_dict(checkpoint['model_state_dict'])
                self.target_net.load_state_dict(checkpoint['model_state_dict'])
                start_episode = checkpoint['epoch']
                self.steps_done = start_episode
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                normalizer.restore_state(os.path.join(os.path.dirname(args.checkpoint_file), 'normalizer_state.pickle'))
                with open(os.path.join(os.path.dirname(args.checkpoint_file), 'results_geom_pomdp.pkl'), 'rb') as file:
                    plot_dict = pickle.load(file)
                    print_variables['durations'] = plot_dict['durations']
                    print_variables['rewards'] = plot_dict['rewards']

        action_space = ActionSpace(dp=0.06, df=10)

        # Main training loop
        for ii in range(start_episode, args.epochs):
            start_time = time.time()
            if args.sim:
                # Create robot, reset simulation and grasp handle
                model = init_model(args.model_path)
                sim = MjSim(model)
                sim.step()
                if args.render:
                    viewer = MjViewer(sim)
                else:
                    viewer = None

                sim_param = SimParameter(sim)
                robot = RobotSim(sim, viewer, sim_param, args.render, self.break_threshold)
                velcro_util = VelcroUtil(robot, sim_param)
                robot.reset_simulation()
                ret = robot.grasp_handle()

                delta = action_space.get_action(self.ACTIONS[3])['delta'][:3]
                target_position = np.add(robot.get_gripper_jpos()[:3], np.array(delta))
                target_pose = np.hstack((target_position, robot.get_gripper_jpos()[3:]))
                robot.move_joint(target_pose, True, self.gripping_force, hap_sample = args.hap_sample)

                if not ret:
                    continue

                # Local memory for current episode
                localMemory = []

                # Get current observation
                hidden_state, cell_state = self.policy_net.init_hidden_states(batch_size=1, device=args.device)
                observation_space = Observation(robot.get_gripper_jpos(),  # 6
                                          velcro_util.break_center(),         # 6
                                          velcro_util.break_norm())  # 12
                broken_so_far = 0

            for t in count():
                if not args.quiet and t % 20 == 0:
                    print("Running training episode: {}, iteration: {}".format(ii, t))

                # Select action
                observation = observation_space.get_state()
                if args.position:
                    observation = observation[6:]
                if args.shear:
                    indices = np.ones(len(observation), dtype=bool)
                    indices[6:166] = False
                    observation = observation[indices]
                if args.force:
                    observation = observation[:166]
                normalizer.observe(observation[:12])
                observation[:12] = normalizer.normalize(observation[:12])
                action, hidden_state, cell_state = self.select_action(observation, hidden_state, cell_state)

                # Perform action
                delta = action_space.get_action(self.ACTIONS[action])['delta'][:3]
                target_position = np.add(robot.get_gripper_jpos()[:3], np.array(delta))
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
                    # reward -= 0.05 * np.linalg.norm(target_position - robot.get_gripper_jpos()[:3]) / np.linalg.norm(delta)

                    # Observe new state
                    observation_space.update(robot.get_gripper_jpos(),  # 6
                                          velcro_util.break_center(),         # 6
                                          velcro_util.break_norm())  # 12

                # Set max number of iterations
                if t >= self.max_iter:
                    done = True

                # Check if done
                if not done and not failure:
                    next_state = observation_space.get_state()
                    if args.position:
                        next_state = next_state[6:]
                    if args.shear:
                        indices = np.ones(len(next_state), dtype=bool)
                        indices[6:166] = False
                        next_state = next_state[indices]
                    if args.force:
                        next_state = next_state[:166]
                    normalizer.observe(observation[:12])
                    observation[:12] = normalizer.normalize(observation[:12])
                else:
                    next_state = None

                # Push new Transition into memory
                localMemory.append(Transition(observation, action, next_state, reward))

                # Optimize the model
                loss = self.optimize_model()
        #        if loss:
        #            print_variables['loss'].append(loss.item())

                # If we are done, reset the model
                if done or failure:
                    self.memory.push(localMemory)
                    if failure:
                        print_variables['durations'].append(self.max_iter)
                    else:
                        print_variables['durations'].append(t)
                    print_variables['rewards'].append(broken_so_far)
                    plot_variables(self.figure, print_variables, "Training POMDP")
                    print("Epoch {} took {}s".format(ii, time.time() - start_time))
                    break

            # Update the target network, every x iterations
            if ii % 10 == 0:
                self.target_net.load_state_dict(self.policy_net.state_dict())

            # Save checkpoints every vew iterations
            if ii % args.save_freq == 0:
                save_path = os.path.join(args.output_dir, 'checkpoint_model_' + str(ii) + '.pth')
                torch.save({
                           'epoch': ii,
                           'model_state_dict': self.target_net.state_dict(),
                           'optimizer_state_dict': self.optimizer.state_dict(),
                           }, save_path)

        # Save normalizer state for inference
        normalizer.save_state(os.path.join(args.output_dir, 'normalizer_state.pickle'))

        print('Training done')
        plt.show()
        return print_variables
