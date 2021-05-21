import os
import sys
import pickle
import argparse
import random
from itertools import count
import numpy as np
from statistics import mean

import torch
from mujoco_py import MjViewer, load_model_from_path, MjSim

from networks.dqn import DRQN
from robot_sim import RobotSim
from sim_param import SimParameter

from utils.gripper_util import change_sim
from utils.normalize import Multimodal_Normalizer
from utils.action_buffer import ActionSpace, TactileObs

NUM_TENDON = 216

def select_action(action):
    sample = random.random()
    p_threshold = 0.05

    if sample > p_threshold:
        return action
    else:
        return random.randrange(0, 6)


def test_network(args, policy_net, normalizer, robot, state_space, performance):
    hidden_state, cell_state = policy_net.init_hidden_states(args.batch_size, args.device)

    action_space = ActionSpace(dp=0.06, df=10)
    ACTIONS = ['left', 'right', 'forward', 'backward', 'up', 'down']
    broken_so_far = 0

    t = 0
    action = 4
    collision = 0
    
    while t < args.max_iter:
        # Observe state and normalize
        state = state_space.get_state()
        normalizer.observe(state)
        state = normalizer.normalize(state)

        torch_observation = torch.from_numpy(state).float().to(args.device).unsqueeze(0)
        model_out = policy_net(torch_observation, batch_size=1, time_step=1, hidden_state=hidden_state, cell_state=cell_state)
        out = model_out[0]
        hidden_state = model_out[1][0]
        cell_state = model_out[1][1]
        # print(out[0])
        action_key = int(torch.argmax(out[0]))
        action = select_action(action_key)
        performance['action_hist'][action] += 1

        # perform action
        delta = action_space.get_action(ACTIONS[action_key])['delta'][:3]
        target_position = np.add(robot.get_gripper_jpos()[:3], np.array(delta))
        target_pose = np.hstack((target_position, robot.get_gripper_jpos()[3:]))
        robot.move_joint(target_pose, True, args.grip, hap_sample=args.hap_sample)
        
        # Get reward
        done, num = robot.update_tendons()
        failure = robot.check_slippage()

        # Observe new state
        state_space.update(robot.get_gripper_xpos(),            # 24
                                 robot.get_all_touch_buffer(args.hap_sample))     # 30x6
        if num > broken_so_far:
            broken_so_far = num

        t += 1

        if done or failure:
            ratio_broken = float(num) / float(NUM_TENDON)
            if ratio_broken < 0.2:
                performance['tendon_hist'][0] += 1
            elif ratio_broken >= 0.2 and ratio_broken < 0.4:
                performance['tendon_hist'][1] += 1
            elif ratio_broken >= 0.4 and ratio_broken < 0.6:
                performance['tendon_hist'][2] += 1
            elif ratio_broken >= 0.6 and ratio_broken < 0.8:
                performance['tendon_hist'][3] += 1
            else:
                performance['tendon_hist'][4] += 1
            performance['num_broken'].append(num)
            if done:
                performance['success'].append(1)
                performance['time'].append(t + 1)
            if failure:
                performance['success'].append(0)
                performance['time'].append(t + 1)
            performance['collision'].append(collision)
            return performance
            break

    ################## exceed max iterations ####################
    performance['success'].append(0)
    performance['time'].append(args.max_iter)
    ratio_broken = float(num) / float(NUM_TENDON)
    performance['num_broken'].append(num)
    if ratio_broken < 0.2:
        performance['tendon_hist'][0] += 1
    elif ratio_broken >= 0.2 and ratio_broken < 0.4:
        performance['tendon_hist'][1] += 1
    elif ratio_broken >= 0.4 and ratio_broken < 0.6:
        performance['tendon_hist'][2] += 1
    elif ratio_broken >= 0.6 and ratio_broken < 0.8:
        performance['tendon_hist'][3] += 1
    else:
        performance['tendon_hist'][4] += 1
    performance['collision'].append(collision)
    return performance

def main(args):
    if not os.path.isdir(args.result_dir):
        os.makedirs(args.result_dir)
    parent = os.path.dirname(os.path.abspath(__file__))
    # load test xml files
    test_file = os.path.join(parent, 'tests/test_xmls/temp_1_{}.pickle'.format(args.case))
    params = pickle.load(open(test_file, 'rb'))
    # params = params[:6]
    if args.shuffle:
        random.shuffle(params)

    num_test = len(params)
    print('                    ++++++++++++++++++++++++++')
    print('                    +++ Now running case {} +++'.format(args.case))
    print('                    ++++++++++++++++++++++++++\n\n')

    policy_net = DRQN(args.indim, args.outdim)
    policy_net.load_state_dict(torch.load(args.weight_path)['policy_net_1'])
    policy_net.eval()

    # load normalizer
    # Setup the state normalizer
    normalizer = Multimodal_Normalizer(num_inputs = args.indim, device=args.device)
    if args.normalizer_file:
        if os.path.exists(args.normalizer_file):
            normalizer.restore_state(args.normalizer_file)

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

    tactile_obs_space = TactileObs(robot.get_gripper_xpos(),            # 24
                         robot.get_all_touch_buffer(args.hap_sample))   # 30 x 6

    performance = {'time':[], 'success':[], 'num_broken':[], 'tendon_hist':[0,0,0,0,0], 'collision':[], 
                    'action_hist': [0,0,0,0,0,0]}
    
    for i in range(num_test):
        velcro_params = params[i]
        geom, origin_offset, euler, radius = velcro_params
        print('\n\nTest {} Velcro parameters are: {}, {}, {}, {}'.format(i, geom, origin_offset, euler, radius))
        change_sim(robot.mj_sim, geom, origin_offset, euler, radius)
        robot.reset_simulation()
        ret = robot.grasp_handle()
        performance = test_network(args, policy_net, normalizer, robot, tactile_obs_space, performance)
        print('Success: {}, time: {}, num_broken: {}, collision:{} '.format(
                performance['success'][-1], performance['time'][-1], performance['num_broken'][-1], performance['collision'][-1]))

    print('Finished opening velcro with haptics test \n')
    success = np.array(performance['success'])
    time = np.array(performance['time'])
    print('Successfully opened the velcro in: {}% of cases'.format(100 * np.sum(success) / len(performance['success'])))
    print('Average time to open: {}'.format(np.average(time[success>0])))
    print('Action histogram for the test is: {}'.format(performance['action_hist']))

    out_fname = 'case{}.txt'.format(args.case)
    with open(os.path.join(args.result_dir, out_fname), 'w+') as f:
        f.write('Time: {}\n'.format(performance['time']))
        f.write('Success: {}\n'.format(performance['success']))
        f.write('Successfully opened the velcro in: {}% of cases\n'.format(100 * np.sum(success) / len(performance['success'])))
        f.write('Average time to open: {}\n'.format(np.average(time[success>0])))
        f.write('Num_broken: {}\n'.format(performance['num_broken']))
        f.write('Tendon histogram: {}\n'.format(performance['tendon_hist']))
        f.write('collision: {}\n'.format(performance['collision']))
        # f.write('high_success: {} low_success: {} '.format(high_success, low_success))
    f.close()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Tactile Test')

    parser.add_argument('--model_path', required=True, help='XML model to load')
    parser.add_argument('--indim', default=234, type=int, help='observation space size')
    parser.add_argument('--outdim', default=6, type=int, help='action space size')
    parser.add_argument('--weight_path', default='/home/jc/logs/haptics_pomdp_2', help='path to load weights')
    parser.add_argument('--normalizer_file', default=None, help='normalizer file to load to resume training')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size for networks')
    parser.add_argument('--break_thresh', default=0.06, type=float, help='velcro breaking threshold')
    parser.add_argument('--device', default='cpu', help='device')
    parser.add_argument('--sim', default=True, help='whether to run in simulation mode or on a real robot')
    parser.add_argument('--render', default=False, type = bool, help='render simulation')
    parser.add_argument('--shuffle', action='store_true', help='shuffle the velcro parameters after loading')
    parser.add_argument('--case', default=1, help='case number')
    parser.add_argument('--hap_sample', default=30, type=int, help='number of haptics samples feedback in each action excution')
    parser.add_argument('--grip', default=400, type=int, help='grip force in each action excution')
    parser.add_argument('--max_iter', default=200, type=int, help='grip force in each action excution')
    parser.add_argument('--result_dir', default='/home/jc/logs/haptics_pomdp_2', help='dir to store results')

    args = parser.parse_args()
    main(args)
