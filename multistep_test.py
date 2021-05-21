import os
import sys
import argparse
import random
import pickle
from itertools import count
import numpy as np
import numpy.linalg as la
from statistics import mean

import torch
from mujoco_py import MjViewer, load_model_from_path, MjSim

from networks.dqn import DQN
from networks.dqn import Geom_DQN
from networks.dqn import DRQN
from networks.tactile_net import TactileNet
from robot_sim import RobotSim
from sim_param import SimParameter

from utils.action_buffer import ActionSpace, TactileObs
from utils.normalize import Multimodal_Normalizer
from utils.velcro_utils import VelcroUtil
from utils.gripper_util import change_sim

NUM_TENDON = 216

def select_action(args, observation, policy_net, tactile_net, hidden_state, cell_state):
    sample = random.random()
    p_threshold = 0.05

    with torch.no_grad():
        tactile_net.eval()
        policy_net.eval()
        torch_obs = torch.from_numpy(observation).float().to(args.device).unsqueeze(0)
        h_tac, c_tac = tactile_net.init_hidden_states(args.device)
        tactile_ft = tactile_net.forward(torch_obs, hidden_state=h_tac, cell_state=c_tac)

        model_out = policy_net(tactile_ft.unsqueeze(1), batch_size=1, time_step=1, hidden_state=hidden_state, cell_state=cell_state)
        out = model_out[0]
        hidden_state = model_out[1][0]
        cell_state = model_out[1][1]
        
        if sample > p_threshold:
            action = int(torch.argmax(out[0]))
            return action, hidden_state, cell_state
        else:
            return random.randrange(0, args.outdim), hidden_state, cell_state


def test_network(args, policy_net, tactile_net, normalizer, robot, obs_space, performance):
    hidden_state, cell_state = policy_net.init_hidden_states(1, args.device)

    action_space = ActionSpace(dp=0.06, df=10)
    ACTIONS = ['left', 'right', 'forward', 'backward', 'up', 'down']
    broken_so_far = 0

    t = 0
    action = 4
    collision = 0
            
    while t < args.max_iter:

        if args.position:
            multistep_obs = np.empty((0, args.indim-6))
        elif args.force:
            multistep_obs = np.empty((0, args.indim-390))
        else:                      
            multistep_obs = np.empty((0, args.indim))

        prev_action = action

        for k in range(args.len_ub):
            # Observe tactile features and stack them
            tactile_obs = obs_space.get_state()
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
            delta = action_space.get_action(ACTIONS[action])['delta'][:3]
            target_position = np.add(robot.get_gripper_jpos()[:3], np.array(delta))
            target_pose = np.hstack((target_position, robot.get_gripper_jpos()[3:]))
            robot.move_joint(target_pose, True, args.grip_force, hap_sample = args.hap_sample)

            # check collision number
            collision += robot.feedback_buffer['collision']

            # Observe new state
            obs_space.update(robot.get_gripper_xpos(),            # 24
                                     robot.get_all_touch_buffer(args.hap_sample))     # 30x6

            displacement = la.norm(robot.get_gripper_jpos()[:3] - current_pos)

            if displacement / 0.06 < 0.7:
                break


        # input stiched multi-step tactile observation into tactile-net to generate tactile feature
        action, hidden_state, cell_state = select_action(args, multistep_obs, policy_net, tactile_net,
                                                         hidden_state, cell_state)

        # record actions in this epoch
        # act_sequence.append(prev_action)
                
        # Get reward
        done, num = robot.update_tendons()
        failure = robot.check_slippage()
        if num > broken_so_far:
            broken_so_far = num
                
        t += k + 1

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

    # Create our policy net and a target net
    policy_net = DRQN(args.ftdim, args.outdim).to(args.device)        
    if args.position:
        tactile_net = TactileNet(args.indim-6, args.ftdim).to(args.device)
    elif args.force:
        tactile_net = TactileNet(args.indim-390, args.ftdim).to(args.device)
    else:
        tactile_net = TactileNet(args.indim, args.ftdim).to(args.device)

    # Setup the state normalizer
    normalizer = Multimodal_Normalizer(num_inputs = args.indim, device=args.device)

    if args.weight_policy:
        checkpoint = torch.load(args.weight_policy)
        policy_net.load_state_dict(checkpoint['policy_net_1'])
    if args.weight_tactile:
        checkpoint = torch.load(args.weight_tactile)
        tactile_net.load_state_dict(checkpoint['tactile_net_1'])
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

    performance = {'time':[], 'success':[], 'num_broken':[], 'tendon_hist':[0,0,0,0,0], 'collision':[]}
    
    for i in range(num_test):
        velcro_params = params[i]
        geom, origin_offset, euler, radius = velcro_params
        print('\n\nTest {} Velcro parameters are: {}, {}, {}, {}'.format(i, geom, origin_offset, euler, radius))
        change_sim(robot.mj_sim, geom, origin_offset, euler, radius)
        robot.reset_simulation()
        ret = robot.grasp_handle()
        performance = test_network(args, policy_net, tactile_net, normalizer, robot, tactile_obs_space, performance)
        print('Success: {}, time: {}, num_broken: {}, collision:{} '.format(
                performance['success'][-1], performance['time'][-1], performance['num_broken'][-1], performance['collision'][-1]))

    print('Finished opening velcro with haptics test \n')
    success = np.array(performance['success'])
    time = np.array(performance['time'])
    print('Successfully opened the velcro in: {}% of cases'.format(100 * np.sum(success) / len(performance['success'])))
    print('Average time to open: {}'.format(np.average(time[success>0])))
    # print('Action histogram for the test is: {}'.format(performance['action_hist']))

    # collision = np.array(performance['collision'])
    # threshold = 3000
    # high_success = float(np.sum(success[collision<threshold])) / float(np.sum(np.ones(num_test)[collision<threshold]))
    # low_success =  float(np.sum(success[collision>threshold])) / float(np.sum(np.ones(num_test)[collision>threshold]))
    # print('high_success: {} low_success: {} '.format(high_success, low_success))

    ablation = 'None'
    if args.position:
        ablation = 'position'
    if args.force:
        ablation = 'force'
    checkpoint = args.weight_policy.split('/')[-1]
    out_fname = 'case{}_{}_{}.txt'.format(args.case, ablation, checkpoint)
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
    ablation = parser.add_mutually_exclusive_group(required=True)
    ablation.add_argument('--none', action='store_true', help='include position, shear and tactile in observation')
    ablation.add_argument('--position', action='store_true', help='remove position from observation')
    ablation.add_argument('--force', action='store_true', help='remove tactile from observation')

    parser.add_argument('--model_path', required=True, help='XML model to load')
    parser.add_argument('--case', required=True, type=int, help='test case to load')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--indim', default=234, type=int, help='observation space size')
    parser.add_argument('--outdim', default=6, type=int, help='action space size')
    parser.add_argument('--ftdim', default=150, type=int, help='tactile feature size')
    parser.add_argument('--break_thresh', default=0.06, type=float, help='velcro breaking threshold')
    parser.add_argument('--max_iter', default=200, type=float, help='max number of iterations per epoch')
    parser.add_argument('--grip_force', default=400, type=float, help='gripping force')
    parser.add_argument('--len_ub', default=15, type=int, help='upper bound of multistep agent takes')
    parser.add_argument('--render', action='store_true', help='turn on rendering')
    parser.add_argument('--shuffle', action='store_true', help='shuffle the velcro parameters after loading')
    parser.add_argument('--weight_policy', default=None, help='checkpoint file to load to resume training')
    parser.add_argument('--normalizer_file', default=None, help='normalizer file to load to resume training')
    parser.add_argument('--weight_tactile', default=None, help='normalizer file to load to resume training')
    parser.add_argument('--hap_sample', default=30, type=int, help='number of haptics samples feedback in each action excution')
    parser.add_argument('--result_dir', default='/home/jc/logs/', help='dir to store results')

    args = parser.parse_args()
    main(args)
