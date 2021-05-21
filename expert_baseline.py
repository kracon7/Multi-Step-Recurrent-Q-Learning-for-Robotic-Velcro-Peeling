import os
import argparse
import time
import math
import random
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from itertools import count
import pickle
import datetime as dt
from mujoco_py import MjViewer, MjSim, load_model_from_path
from robot_sim import RobotSim
from sim_param import SimParameter
from utils.action_buffer import ActionSpace, Observation
from utils.visualization import plot_variables
from utils.gripper_util import init_model, change_sim
from utils.velcro_utils import VelcroUtil

VELCRO_PARAMS =[['cylinder', [0., 0, 0.0], [0.,           0.,           0.], 0.6],
                ['cylinder', [0., 0.2, 0.0], [-0.35   ,    0.,  0.], 0.6],
                ['cylinder', [0., -0.2, 0.0], [0., -0.35, 0.], 0.6]]
ACTIONS = ['left', 'right', 'forward', 'backward', 'up', 'down']
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
    action_space = ActionSpace(dp=0.06, df=10)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.clear()

    for item in VELCRO_PARAMS:
        geom_type, origin_offset, euler, radius = item
        print('Velcro parameters are: {}, {}, {}, {}'.format(geom_type, origin_offset, euler, radius))
        change_sim(robot.mj_sim, geom_type, origin_offset, euler, radius)
        robot.reset_simulation()
        ret = robot.grasp_handle()
        broken_so_far = 0

        ax.clear()

        
        for i in range(args.max_iter):
            # check velcro breakline geometry
            norms = velcro_util.break_norm()
            centers = velcro_util.break_center()
            fl_center = centers[:3]
            fs_center = centers[3:]
            fl_norm = norms[:3]
            fs_norm = norms[3:6]
            fs_center = centers[3:]
            break_dir_norm = norms[6:9]

            action_direction = args.act_mag*(-0.5 * fl_norm + 0.5 * break_dir_norm)

            gripper_pose = robot.get_gripper_jpos()[:3]

            # Perform action
            target_position = np.add(robot.get_gripper_jpos()[:3], action_direction)
            target_pose = np.hstack((target_position, robot.get_gripper_jpos()[3:]))
            robot.move_joint(target_pose, True, 300, hap_sample=30)


            ax.scatter(fl_center[0], fl_center[1], fl_center[2], c='r', marker='o')
            ax.scatter(fs_center[0], fs_center[1], fs_center[2], c='r', marker='o')
            X, Y, Z = lines(fl_center, fl_norm)
            ax.plot(X, Y, Z, c='b')
            X, Y, Z = lines(fs_center, fs_norm)
            ax.plot(X, Y, Z, c='b')

            # plot gripper position and action direction
            ax.scatter(gripper_pose[0], gripper_pose[1], gripper_pose[2], c='r', marker='v')
            X, Y, Z = lines(gripper_pose, action_direction/args.act_mag)
            ax.plot(X, Y, Z, c='g')
            plt.pause(0.001)

            # check tendons and slippage
            done, num = robot.update_tendons()
            failure = robot.check_slippage()
            if num > broken_so_far:
                broken_so_far = num

            if done or failure:
                break


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='PyTorch Tactile Training')
    parser.add_argument('--model_path', default='./models/flat_velcro.xml', help='XML model to load')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('--max_iter', default=200, type=float, help='max number of iterations per epoch')
    parser.add_argument('--output_dir', default='.', help='path where to save')
    parser.add_argument('--render', action='store_true', help='turn on rendering')
    parser.add_argument('--break_thresh', default=0.06, type=float, help='velcro breaking threshold')
    parser.add_argument('--act_mag', default=0.06, type=float, help='action magnitude')
        
    args = parser.parse_args()
    main(args)

