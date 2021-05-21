import os
import sys
import argparse
import random
from itertools import count
import numpy as np
from statistics import mean

import torch
from mujoco_py import MjViewer, load_model_from_path, MjSim

from networks.dqn import Geom_DQN
from robot_sim import RobotSim
from sim_param import SimParameter

from utils.normalize import Normalizer
from utils.normalize import Geom_Normalizer
from utils.action_buffer import ActionSpace, Observation
from utils.velcro_utils import VelcroUtil
from utils.gripper_util import init_for_test

ACTIONS = ['left', 'right', 'forward', 'backward', 'up', 'down']

def select_action(policy_net, state):
    sample = random.random()
    p_threshold = 0.1

    if sample > p_threshold:
        with torch.no_grad():
            policy_net.eval()
            torch_state = torch.from_numpy(state).float().to(args.device)
            action = policy_net(torch_state.unsqueeze(0)).max(1)[1]
            return action.item()
    else:
        return random.randrange(6)

def test_network(args, policy_net, normalizer, performance, robot, velcro_util, max_iterations):
    
    # Get current state
    state_space = Observation(robot.get_gripper_jpos(),  # 6
                                          velcro_util.break_center(),         # 6
                                          velcro_util.break_norm())  # 12

    action_space = ActionSpace(dp=0.06, df=10)
    
    broken_so_far = 0
    
    for t in range(max_iterations):
        # Observe state and normalize
        state = state_space.get_state()
        normalizer.observe(state[:12])
        state[:12] = normalizer.normalize(state[:12])
        action = select_action(policy_net, state)
        performance['action_hist'][action] += 1

        # perform action
        delta = action_space.get_action(ACTIONS[action])['delta'][:3]
        target_position = np.add(robot.get_gripper_jpos()[:3], np.array(delta))
        target_pose = np.hstack((target_position, robot.get_gripper_jpos()[3:]))
        robot.move_joint(target_pose, True, args.grip_force)
        
        # Get reward
        done, num = robot.update_tendons()
        failure = robot.check_slippage()
        if num > broken_so_far:
            broken_so_far = num

        if not done and not failure:
            # Observe new state
            state_space.update(robot.get_gripper_jpos(),  # 6
                                          velcro_util.break_center(),         # 6
                                          velcro_util.break_norm())  # 12
        else:
            if done:
                performance['success'].append(1)
                performance['time'].append(t + 1)
            if failure:
                performance['success'].append(0)
                performance['time'].append(t + 1)
            return performance
            break
    # exceed max iterations
    performance['success'].append(0)
    performance['time'].append(max_iterations)
    return performance


def main(args):

    policy_net = Geom_DQN(args.indim, args.outdim).to(args.device)
    policy_net.load_state_dict(torch.load(args.checkpoint_file)['policy_net_1'])
    policy_net.eval()

    normalizer = Geom_Normalizer(args.indim, device=args.device)
    normalizer.restore_state(args.normalizer_file)

    max_iterations = args.max_iter
    performance = {'time':[], 'success':[], 'action_hist':[0,0,0,0,0,0]}
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
    velcro_util = VelcroUtil(robot, robot.mj_sim_param)

    '''
    Test case 1
    '''
    num_case = 50
    for i in range(num_case):
        velcro_params = init_for_test(robot.mj_sim)
        robot.reset_simulation()
        ret = robot.grasp_handle()
        performance = test_network(args, policy_net, normalizer, performance, robot, velcro_util, max_iterations)
        print(performance)

        print('Finished opening velcro with haptics test \n')
        success = np.array(performance['success'])
        time = np.array(performance['time'])
        print('Successfully opened the velcro in: {}% of cases'.format(100 * np.sum(success) / len(performance['success'])))
        print('Average time to open: {}'.format(np.average(time[success>0])))
        print('Action histogram for the test is: {}'.format(performance['action_hist']))

    out_fname = 'case{}.txt'.format(args.case)
    with open(os.path.join(args.result_dir, out_fname), 'w+') as f:
        f.write(performance['time'])
        f.write(performance['success'])
        f.write(performance['action_hist'])
    f.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Tactile Test')
    parser.add_argument('--model_path', required=True, help='XML model to load')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--indim', default=24, type=int, help='observation space size')
    parser.add_argument('--outdim', default=6, type=int, help='action space size')
    parser.add_argument('--break_thresh', default=0.06, type=float, help='velcro breaking threshold')
    parser.add_argument('--max_iter', default=200, type=float, help='max number of iterations per epoch')
    parser.add_argument('--grip_force', default=300, type=float, help='gripping force')
    parser.add_argument('--output_dir', default='.', help='path where to save')
    parser.add_argument('--quiet', action='store_true', help='wether to print episodes or not')
    parser.add_argument('--render', default=False, type=bool, help='turn on rendering')
    parser.add_argument('--checkpoint_file', default=None, help='checkpoint file to load to resume training')
    parser.add_argument('--normalizer_file', default=None, help='normalizer file to load to resume training')
    parser.add_argument('--case', default=1, help='case number')

    args = parser.parse_args()
    main(args)
