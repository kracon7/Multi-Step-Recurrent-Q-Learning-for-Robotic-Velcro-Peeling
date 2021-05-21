import os
import time
import math
import random
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
import datetime as dt

import torch
import torch.optim as optim
import torch.nn.functional as F
from mujoco_py import MjViewer, load_model_from_path, MjSim
from networks.dqn import DRQN
from robot_sim import RobotSim
from sim_param import SimParameter
from utils.action_buffer import ActionSpace, TactileObs
from utils.memory import RecurrentMemory, Transition
from utils.normalize import Multimodal_Normalizer
from utils.visualization import plot_variables
from utils.gripper_util import init_model, norm_img

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
        self.P_DECAY = 600
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

        with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            self.policy_net_1.eval()
            torch_observation = torch.from_numpy(observation).float().to(self.args.device).unsqueeze(0)
            model_out = self.policy_net_1(torch_observation, batch_size=1, time_step=1, hidden_state=hidden_state, cell_state=cell_state)
            out = model_out[0]
            hidden_state = model_out[1][0]
            cell_state = model_out[1][1]
            self.policy_net_1.train()

            if sample > p_threshold:
                action = int(torch.argmax(out[0]))
                return action, hidden_state, cell_state
            else:
                return random.randrange(0, self.args.outdim), hidden_state, cell_state

    def optimize_model(self):
        args = self.args
        if len(self.memory) < (args.batch_size):
            return

        hidden_batch_1, cell_batch_1 = self.policy_net_1.init_hidden_states(args.batch_size, args.device)
        hidden_batch_2, cell_batch_2 = self.policy_net_2.init_hidden_states(args.batch_size, args.device)
        hidden_batch_3, cell_batch_3 = self.policy_net_3.init_hidden_states(args.batch_size, args.device)
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
        Q_s_1, _ = self.policy_net_1.forward(current_states,
                                         batch_size=args.batch_size,
                                         time_step=args.time_step,
                                         hidden_state=hidden_batch_1,
                                         cell_state=cell_batch_1)
        Q_s_a_1 = Q_s_1.gather(dim=1, index=actions[:, args.time_step - 1].unsqueeze(dim=1)).squeeze(dim=1)

        Q_s_2, _ = self.policy_net_2.forward(current_states,
                                         batch_size=args.batch_size,
                                         time_step=args.time_step,
                                         hidden_state=hidden_batch_2,
                                         cell_state=cell_batch_2)
        Q_s_a_2 = Q_s_2.gather(dim=1, index=actions[:, args.time_step - 1].unsqueeze(dim=1)).squeeze(dim=1)

        Q_s_3, _ = self.policy_net_3.forward(current_states,
                                         batch_size=args.batch_size,
                                         time_step=args.time_step,
                                         hidden_state=hidden_batch_3,
                                         cell_state=cell_batch_3)
        Q_s_a_3 = Q_s_3.gather(dim=1, index=actions[:, args.time_step - 1].unsqueeze(dim=1)).squeeze(dim=1)



        Q_next_1, _ = self.policy_net_1.forward(next_states,
                                            batch_size=args.batch_size,
                                            time_step=args.time_step,
                                            hidden_state=hidden_batch_1,
                                            cell_state=cell_batch_1)
        Q_next_max_1 = Q_next_1.detach().max(dim=1)[0]

        Q_next_2, _ = self.policy_net_2.forward(next_states,
                                            batch_size=args.batch_size,
                                            time_step=args.time_step,
                                            hidden_state=hidden_batch_2,
                                            cell_state=cell_batch_2)
        Q_next_max_2 = Q_next_2.detach().max(dim=1)[0]

        Q_next_3, _ = self.policy_net_3.forward(next_states,
                                            batch_size=args.batch_size,
                                            time_step=args.time_step,
                                            hidden_state=hidden_batch_3,
                                            cell_state=cell_batch_3)
        Q_next_max_3 = Q_next_3.detach().max(dim=1)[0]

        Q_next_max = torch.min(torch.min(Q_next_max_1, Q_next_max_2), Q_next_max_3)

        # Compute the expected Q values
        target_values = rewards[:, args.time_step - 1] + (args.gamma * Q_next_max)

        # Compute Huber loss
        loss_1 = F.smooth_l1_loss(Q_s_a_1, target_values)
        loss_2 = F.smooth_l1_loss(Q_s_a_2, target_values)
        loss_3 = F.smooth_l1_loss(Q_s_a_3, target_values)

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
        self.policy_net_1 = DRQN(args.indim, args.outdim).to(args.device)        
        self.policy_net_2 = DRQN(args.indim, args.outdim).to(args.device)        
        self.policy_net_3 = DRQN(args.indim, args.outdim).to(args.device)

        # Set up the optimizer
        self.optimizer_1 = optim.RMSprop(self.policy_net_1.parameters())
        self.optimizer_2 = optim.RMSprop(self.policy_net_2.parameters())
        self.optimizer_3 = optim.RMSprop(self.policy_net_3.parameters())
        self.memory = RecurrentMemory(800)
        self.steps_done = 0

        # Setup the state normalizer
        normalizer = Multimodal_Normalizer(num_inputs = args.indim, device=args.device)

        print_variables = {'durations': [], 'rewards': [], 'loss': []}
        start_episode = 0
        if args.checkpoint_file:
            if os.path.exists(args.checkpoint_file):
                checkpoint = torch.load(args.checkpoint_file)
                self.policy_net_1.load_state_dict(checkpoint['policy_net_1'])
                self.policy_net_2.load_state_dict(checkpoint['policy_net_2'])
                self.policy_net_3.load_state_dict(checkpoint['policy_net_3'])
                self.optimizer_1.load_state_dict(checkpoint['optimizer_1'])
                self.optimizer_2.load_state_dict(checkpoint['optimizer_2'])
                self.optimizer_3.load_state_dict(checkpoint['optimizer_3'])
                start_episode = checkpoint['epochs']
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
        
        # Main training loop
        for ii in range(start_episode, args.epochs):
            start_time = time.time()
            self.steps_done += 1
            act_sequence =[]
            if args.sim:
                sim_params = init_model(robot.mj_sim)
                robot.reset_simulation()
                ret = robot.grasp_handle()
                if not ret:
                    continue

                # Local memory for current episode
                localMemory = []

                # Get current observation
                hidden_state_1, cell_state_1 = self.policy_net_1.init_hidden_states(batch_size=1, device=args.device)
                hidden_state_2, cell_state_2 = self.policy_net_2.init_hidden_states(batch_size=1, device=args.device)
                hidden_state_3, cell_state_3 = self.policy_net_3.init_hidden_states(batch_size=1, device=args.device)
                observation_space = TactileObs(robot.get_gripper_xpos(),            # 24
                             					robot.get_all_touch_buffer(args.hap_sample))   # 30 x 7
                broken_so_far = 0

            for t in count():
                if not args.quiet and t % 50 == 0:
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
                normalizer.observe(observation)
                observation = normalizer.normalize(observation)
                action, hidden_state_1, cell_state_1 = self.select_action(observation, hidden_state_1, cell_state_1)
                
                # record actions in this epoch
                act_sequence.append(action)

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
                    observation_space.update(robot.get_gripper_xpos(),            # 24
                                             robot.get_all_touch_buffer(args.hap_sample))     # 30x7

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
                    normalizer.observe(next_state)
                    next_state = normalizer.normalize(next_state)
                else:
                    next_state = None

                # Push new Transition into memory
                localMemory.append(Transition(observation, action, next_state, reward))

                # Optimize the model
                if t%10 == 0:
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
                    print("Model parameters: {}".format(sim_params))
                    print("Actions in this epoch are: {}".format(act_sequence))
                    print("Epoch {} took {}s, total number broken: {}\n\n".format(ii, time.time() - start_time, broken_so_far))

                    break

            # Save checkpoints every vew iterations
            if ii % args.save_freq == 0:
                save_path = os.path.join(output_dir, 'checkpoint_model_' + str(ii) + '.pth')
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

            self.memory.save_memory(os.path.join(output_dir, 'memory.pickle'))


        if args.savefig_path:
            now = dt.datetime.now()
            self.figure[0].savefig(args.savefig_path+'{}_{}_{}.png'.format(now.month, now.day, now.hour), format='png')

        print('Training done')
        plt.show()
        return print_variables
