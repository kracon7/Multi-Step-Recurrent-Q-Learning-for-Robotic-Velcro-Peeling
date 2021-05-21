import os
import argparse
import time
import math
import random
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import count
import pickle
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import datetime as dt
from mujoco_py import MjViewer, MjSim, load_model_from_path
from robot_sim import RobotSim
from sim_param import SimParameter
from utils.action_buffer import ActionSpace, Observation, TactileObs
from utils.visualization import plot_variables
from utils.gripper_util import init_model, change_sim, norm_img, norm_depth
from utils.velcro_utils import VelcroUtil
from utils.normalize import Normalizer, Multimodal_Normalizer

class Net(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(Net, self).__init__()
        self.input_size = in_dim
        self.output_size = out_dim
        n_filters = 1024
        self.model = nn.Sequential(nn.Linear(in_dim, n_filters),
                                  nn.ReLU(),
                                  nn.Linear(n_filters, n_filters // 2),
                                  nn.ReLU(),
                                  nn.Linear(n_filters // 2, n_filters // 4),
                                  nn.ReLU(),
                                  nn.Linear(n_filters // 4, n_filters // 4),
                                  nn.ReLU(),
                                  nn.Linear(n_filters // 4, out_dim))

    def forward(self, x):
        out = self.model(x)
        return out


VELCRO_PARAMS =[['cylinder', [0., 0, 0.0], [0.75,           0.75,           0.], 0.6],
                ['cylinder', [0., 0.2, 0.0], [-0.75   ,    0.,  0.], 0.6],
                ['cylinder', [0., -0.2, 0.0], [0., -0.75, 0.], 0.6]]
action_vec = 0.06 * np.array([  [-1., 0., 0.],
                                [ 1., 0., 0.],
                                [ 0., 1., 0.],
                                [ 0.,-1., 0.],
                                [ 0., 0., 1.],
                                [ 0., 0.,-1.]])
ACTIONS = ['left', 'right', 'forward', 'backward', 'up', 'down']
NUM_TENDON = 216
plt.ion()

def lines(start_point, norm):
    X = np.zeros(2)
    Y = np.zeros(2)
    Z = np.zeros(2)
    X[0] = start_point[0]
    X[1] = start_point[0] + norm[0]
    Y[0] = start_point[1]
    Y[1] = start_point[1] + norm[1]
    Z[0] = start_point[2]
    Z[1] = start_point[2] + norm[2]
    return X, Y, Z


def main(args):
    if not os.path.isdir(args.result_dir):
        os.makedirs(args.result_dir)

    net = Net(args.indim, args.outdim).to(args.device)
    if os.path.exists(args.weight_net):
        checkpoint = torch.load(args.weight_net)
        net.load_state_dict(checkpoint['net'])

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
    tactile_obs_space = TactileObs(robot.get_gripper_jpos(),            # 6
                             robot.get_all_touch_buffer(args.hap_sample))   # 30 x 12
    normalizer = Multimodal_Normalizer(num_inputs = args.indim, device=args.device)
    normalizer.restore_state(args.normalizer_file)

    # load all velcro parameters
    # model_dir = os.path.dirname(args.model_path)
    # param_path = os.path.join(model_dir, 'uniform_sample.pkl')
    param_path = '/home/jc/research/corl2019_learningHaptics/tests/test_xmls/case_{}.pickle'.format(args.case)
    velcro_params = pickle.load(open(param_path, 'rb'))
    if args.shuffle:
        random.shuffle(velcro_params)

    action_space = ActionSpace(dp=0.06, df=10)
    performance = {'time':[], 'success':[], 'num_broken':[], 'tendon_hist':[0,0,0,0,0]}

    for n, item in enumerate(velcro_params):
        geom_type, origin_offset, euler, radius = item
        print('\n\nTest {} Velcro parameters are: {}, {}, {}, {}'.format(n, geom_type, origin_offset, euler, radius))
        change_sim(robot.mj_sim, geom_type, origin_offset, euler, radius)
        robot.reset_simulation()
        ret = robot.grasp_handle()
        broken_so_far = 0

        # ax.clear()
        tactile_sequence = []

        for t in range(args.max_iter):
            # take tactile observation and normalize it
            tactile_obs_space.update(robot.get_gripper_jpos(),            # 6
                                     robot.get_all_touch_buffer(args.hap_sample))     # 30x12
            tactile_obs = tactile_obs_space.get_state()
            tactile_obs = normalizer.normalize(tactile_obs)
            tactile_sequence.append(tactile_obs[:366])

            if len(tactile_sequence)<10:
                continue
            elif len(tactile_sequence) == 11:
                tactile_sequence.pop(0)
            elif len(tactile_sequence) > 11:
                raise ValueError('tactile_sequence length larger than 11')

            torch_input = torch.from_numpy(np.stack(tactile_sequence).flatten()).float().to(args.device).unsqueeze(0)

            pred = net.forward(torch_input).detach().cpu()
            fl_norm = pred[0][:3].numpy()
            break_dir_norm = pred[0][3:].numpy()
            # normalize these vectors
            fl_norm = fl_norm / la.norm(fl_norm)
            break_dir_norm = break_dir_norm / la.norm(break_dir_norm)

            ################ choose action and get action direction vector ################
            action_direction = args.act_mag*(-0.5 * fl_norm + 0.5 * break_dir_norm)
            action_key = (action_vec @ action_direction).argmax()
            print(action_key)
            action_direction = action_space.get_action(ACTIONS[action_key])['delta'][:3]
                    

            gripper_pose = robot.get_gripper_jpos()[:3]

            # Perform action
            target_position = np.add(robot.get_gripper_jpos()[:3], action_direction)
            target_pose = np.hstack((target_position, robot.get_gripper_jpos()[3:]))
            robot.move_joint(target_pose, True, 300, hap_sample=30)

            # check tendons and slippage
            done, num = robot.update_tendons()
            failure = robot.check_slippage()
            if num > broken_so_far:
                broken_so_far = num

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
                break

            if t == args.max_iter-1:

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

        # print episode performance
        print('Success: {}, time: {}, num_broken: {} '.format(performance['success'][-1], performance['time'][-1], performance['num_broken'][-1]))


    print('Finished opening velcro with haptics test \n')
    success = np.array(performance['success'])
    time = np.array(performance['time'])
    print('Successfully opened the velcro in: {}% of cases'.format(100 * np.sum(success) / len(performance['success'])))
    print('Average time to open: {}'.format(np.average(time[success>0])))
    
    out_fname = 'vision_case{}.txt'.format(args.case)
    with open(os.path.join(args.result_dir, out_fname), 'w+') as f:
        f.write('Time: {}\n'.format(performance['time']))
        f.write('Success: {}\n'.format(performance['success']))
        f.write('Successfully opened the velcro in: {}% of cases\n'.format(100 * np.sum(success) / len(performance['success'])))
        f.write('Average time to open: {}\n'.format(np.average(time[success>0])))
        f.write('Num_broken: {}\n'.format(performance['num_broken']))
        f.write('Tendon histogram: {}\n'.format(performance['tendon_hist']))
    f.close()
        

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Tactile Training')
    parser.add_argument('--model_path', default='./models/flat_velcro.xml', help='XML model to load')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--max_iter', default=200, type=int, help='max number of iterations per epoch')
    parser.add_argument('--render', action='store_true', help='turn on rendering')
    parser.add_argument('--shuffle', action='store_true', help='shuffle the velcro parameters after loading')
    parser.add_argument('--break_thresh', default=0.06, type=float, help='velcro breaking threshold')
    parser.add_argument('--normalizer_file', default='/home/jc/logs/tactile_normalizer.pickle', help='normalizer file to load to resume training')
    parser.add_argument('--indim', default=3660, type=int, help='observation space size')
    parser.add_argument('--outdim', default=6, type=int, help='action space size')
    parser.add_argument('--act_mag', default=0.06, type=float, help='robot action magnitude')
    parser.add_argument('--hap_sample', default=30, type=int, help='number of haptics samples feedback in each action excution')
    parser.add_argument('--case', required=True, type=int, help='test case to load')
    parser.add_argument('--weight_net', default=None, help='checkpoint file to load to resume training')
    parser.add_argument('--result_dir', default='/home/jc/logs/', help='dir to store results')
    
        
    args = parser.parse_args()
    main(args)

