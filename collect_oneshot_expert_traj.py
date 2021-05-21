import os
import sys
import argparse
import random
import pickle
from itertools import count
import numpy as np
from statistics import mean

import torch
from mujoco_py import MjViewer, load_model_from_path, MjSim

from networks.dqn import Geom_DQN
from robot_sim import RobotSim
from sim_param import SimParameter
import matplotlib.pyplot as plt

from utils.normalize import Normalizer, Multimodal_Normalizer, Geom_Normalizer
from utils.action_buffer import ActionSpace, Observation, TactileObs
from utils.velcro_utils import VelcroUtil
from utils.gripper_util import change_sim, norm_img, norm_depth

plt.ion()

def write_results(path, results):
    f = open(path, 'wb')
    pickle.dump(results, f)
    f.close()

class TrajLogger:
    def __init__(self, args, robot, velcro_util, policy_net, normalizer, tactile_normalizer):
        self.args = args
        self.ACTIONS = ['left', 'right', 'forward', 'backward', 'up', 'down']
        self.robot = robot
        self.velcro_util = velcro_util
        self.policy_net = policy_net
        self.normalizer = normalizer
        self.tactile_normalizer = tactile_normalizer

    def select_action(self, policy_net, state):
        sample = random.random()
        p_threshold = self.args.p_thresh

        if sample > p_threshold:
            with torch.no_grad():
                self.policy_net.eval()
                torch_state = torch.from_numpy(state).float().to(args.device)
                action = self.policy_net(torch_state.unsqueeze(0)).max(1)[1]
                return action.item()
        else:
            return random.randrange(6)

    def test_network(self, performance):
        args = self.args
        max_iterations = args.max_iter

        # Get current state
        state_space = Observation(  self.robot.get_gripper_jpos(),  # 6
                                    self.velcro_util.break_center(),         # 6
                                    self.velcro_util.break_norm())  # 12

        tactile_obs_space = TactileObs( self.robot.get_gripper_jpos(),            # 6
                                        self.robot.get_all_touch_buffer(args.hap_sample))   # 30 x 12

        action_space = ActionSpace(dp=0.06, df=10)
        
        broken_so_far = 0
        expert_traj = {'image': None, 'tactile': [], 'action': [], 'position': []}

        img = self.robot.get_img(args.img_w, args.img_h, 'c1', args.depth)
        if args.depth:
            depth = norm_depth(img[1])
            img = norm_img(img[0])
            img_norm = np.empty((4, args.img_w, args.img_h))
            img_norm[:3,:,:] = img
            img_norm[3,:,:] = depth
        else:
            img_norm = norm_img(img)
        expert_traj['image'] = img_norm

        
        for t in range(max_iterations):
            # Observe state and normalize
            state = state_space.get_state()
            # self.normalizer.observe(state[:12])
            state[:12] = self.normalizer.normalize(state[:12])
            action = self.select_action(self.policy_net, state)
            performance['action_hist'][action] += 1

            # record tactile and visual observation, corresponding action
            tactile_obs_space.update(self.robot.get_gripper_jpos(),            # 6
                                        self.robot.get_all_touch_buffer(args.hap_sample))
            tactile_obs = tactile_obs_space.get_state()
            tactile_obs = self.tactile_normalizer.normalize(tactile_obs)
            
            expert_traj['tactile'].append(tactile_obs.tolist())
            expert_traj['action'].append(action)
            expert_traj['position'].append(self.robot.get_gripper_jpos()[:3].tolist())

            # perform action
            delta = action_space.get_action(self.ACTIONS[action])['delta'][:3]
            target_position = np.add(self.robot.get_gripper_jpos()[:3], np.array(delta))
            target_pose = np.hstack((target_position, self.robot.get_gripper_jpos()[3:]))
            self.robot.move_joint(target_pose, True, args.grip_force, hap_sample = args.hap_sample)
            
            # Get reward
            done, num = self.robot.update_tendons()
            failure = self.robot.check_slippage()
            if num > broken_so_far:
                broken_so_far = num

            if not done and not failure:
                # Observe new state
                state_space.update( self.robot.get_gripper_jpos(),  # 6
                                    self.velcro_util.break_center(),         # 6
                                    self.velcro_util.break_norm())  # 12
            else:
                if done:
                    performance['success'].append(1)
                    performance['time'].append(t + 1)
                if failure:
                    performance['success'].append(0)
                    performance['time'].append(t + 1)
                return performance, expert_traj
                break
        # exceed max iterations
        performance['success'].append(0)
        performance['time'].append(max_iterations)
        return performance, expert_traj

def main(args):

    policy_net = Geom_DQN(args.indim, args.outdim).to(args.device)
    policy_net.load_state_dict(torch.load(args.weight_expert)['policy_net_1'])
    policy_net.eval()

    normalizer = Geom_Normalizer(args.indim, device=args.device)
    normalizer.restore_state(args.norm_expert)

    tactile_normalizer = Multimodal_Normalizer(num_inputs = args.tactile_indim, device=args.device)
    tactile_normalizer.restore_state(args.tactile_normalizer)

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

    # load all velcro parameters
    model_dir = os.path.dirname(args.model_path)
    param_path = os.path.join(model_dir, 'uniform_sample.pkl')
    velcro_params = pickle.load(open(param_path, 'rb'))

    robot = RobotSim(sim, viewer, sim_param, args.render, args.break_thresh)
    velcro_util = VelcroUtil(robot, robot.mj_sim_param)

    traj_logger = TrajLogger(args, robot, velcro_util, policy_net, normalizer, tactile_normalizer)

    all_trajectories = []
    all_success = [None for i in range(len(velcro_params))]
    all_time = [None for i in range(len(velcro_params))]

    for i in range(len(velcro_params)):
        geom_type, origin_offset, euler, radius = velcro_params[i]
        change_sim(robot.mj_sim, geom_type, origin_offset, euler, radius)
        performance = {'time':[], 'success':[], 'action_hist':[0,0,0,0,0,0]}
        min_time = args.max_iter
        all_trajectories.append(None)
    
        for j in range(args.num_try):
            robot.reset_simulation()
            ret = robot.grasp_handle()
            performance, expert_traj = traj_logger.test_network(performance)
            if performance['time'][-1] <= min_time:
                all_trajectories[i] = expert_traj
                min_time = performance['time'][-1]
                all_success[i] = performance['success'][j]
                all_time[i] = performance['time'][-1]


        print('\n\nFinished trajectory {}'.format(i))
        print('Velcro parameters are:{} {} {} {}'.format(geom_type, origin_offset, euler, radius))
        print(performance)
        success = np.array(performance['success'])
        time = np.array(performance['time'])
        print('Successfully opened the velcro in: {}% of cases'.format(100 * np.sum(success) / len(performance['success'])))
        print('Average time to open: {}'.format(np.average(time[success>0])))
        print('Action histogram for the test is: {}'.format(performance['action_hist']))

    print('\nCollected {} successful expert trajectories in total'.format(np.sum(np.array(all_success))))
    print('Total success and time: {}, {}'.format(all_success, all_time))
    output = {'args': args, 'traj': all_trajectories, 'success': all_success, 'all_time': all_time}
    output_path = args.result_dir + 'oneshot_expert_traj.pkl'
    write_results(output_path, output)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Tactile Test')
    parser.add_argument('--model_path', required=True, help='XML model to load')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--indim', default=24, type=int, help='observation space size')
    parser.add_argument('--outdim', default=6, type=int, help='action space size')
    parser.add_argument('--tactile_indim', default=396, type=int, help='tactile input size')
    parser.add_argument('--ftdim', default=100, type=int, help='action space size')
    parser.add_argument('--break_thresh', default=0.06, type=float, help='velcro breaking threshold')
    parser.add_argument('--max_iter', default=200, type=float, help='max number of iterations per epoch')
    parser.add_argument('--grip_force', default=300, type=float, help='gripping force')
    parser.add_argument('--result_dir', default='.', help='path where to save')
    parser.add_argument('--quiet', action='store_true', help='wether to print episodes or not')
    parser.add_argument('--render', default=False, type=bool, help='turn on rendering')
    parser.add_argument('--weight_expert', default=None, help='checkpoint file to load to resume training')
    parser.add_argument('--norm_expert', default=None, help='normalizer file to load to resume training')
    parser.add_argument('--tactile_normalizer', default=None, help='tactile normalizer file to load to resume training')
    parser.add_argument('--num_try', default=5, type=int, help='case number')
    parser.add_argument('--p_thresh', default=0.1, type=float, help='randomness threshold for action selection')
    parser.add_argument('--img_w', default=200, type=int, help='observation image width')
    parser.add_argument('--img_h', default=200, type=int, help='observation image height')
    parser.add_argument('--depth', default=True, type=bool, help='use depth from rendering as input')
    parser.add_argument('--hap_sample', default=30, type=int, help='number of haptics samples feedback in each action excution')

    args = parser.parse_args()
    main(args)
