import os
import sys
import argparse
import pickle

# baseline demo for peel velcro with paraGripper
import numpy as np
import scipy.linalg as la
import matplotlib.pyplot as plt
import math
from math import sin, cos
from mujoco_py import MjViewer, load_model_from_path, MjSim

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent)
from robot_sim import RobotSim
from sim_param import SimParameter
from utils import gripper_util
from utils.velcro_utils import VelcroUtil
from utils.action_buffer import get_action_sequence


def run_openloop(robot, dP, args):
    max_time = args.max_iterations
    result = {'done':False, 'time':max_time} 

    robot.reset_simulation()

    robot_pos = np.empty((0, 3))
    robot_shear = np.empty((0,8))
    robot_touch = np.empty((0,2))

    ######################################
    ## move gripper to grasp the handle ##
    ret = robot.grasp_handle()
    if ret:
        robot.checkTendons = True

        feedback = {'touch': [], 'pose': []}


        for k in range(max_time):
            currentGripperJointValues = robot.get_gripper_jpos()
            jointValues_target = currentGripperJointValues + np.array([dP[0], dP[1], dP[2], 0, 0, 0])

            robot.move_joint(jointValues_target, fingerClose=True, fingerCloseForce=args.grip, hap_sample=args.hap_sample)
            tactile_fb = robot.feedback_buffer["main_touch"]
            tool_pose_fb = robot.feedback_buffer["tool_pose"]

            feedback['touch'].append(tactile_fb)
            feedback['pose'].append(tool_pose_fb)

            # check tendons
            done, num_broken = robot.update_tendons()
            
            # check slippage
            slippage = robot.check_slippage()

            if done:
                result['done'] = True 
                result['time'] = k
                # plot_variables((fig,ax1,ax2,ax3), (robot_fpos, robot_shear, robot_maintouch))
                return result, feedback
            if slippage:
                # plot_variables((fig,ax1,ax2,ax3), (robot_fpos, robot_shear, robot_maintouch))
                return result, feedback
        return result, feedback
    else:
        return result, None


def main(args):
    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(parent, 'models/flat_velcro.xml')
    model = load_model_from_path(model_path)
    sim = MjSim(model)
    viewer = MjViewer(sim)
    sim_param = SimParameter(sim)
    robot = RobotSim(sim, viewer, sim_param, args.render, args.break_thresh)

    result = []

    num_tests = 4

    for k in range(num_tests):
        robot.reset_simulation()
        tendon_idx = sim_param.velcro_tendon_id[-(k)*36:0]
        sim.model.tendon_stiffness[tendon_idx] = 0
        # sim.model.body_pos[sim.model._body_name2id['handle']] = np.array([-0.06* (k+1), 0.05, 0.02]) 
        # sim.model.tendon_range[sim_param.handle_tendon_id,1] = (k+1) *np.array([.061, .061, .061, .061, .061, .061])

        N = 10
        alpha = np.arange(1/N, 1, 1/N)
        for i in range(N-1):
            dP = np.array([cos(alpha[i] * np.pi), 0, sin(alpha[i] * np.pi)]) * args.act_mag
            ret, feedback = run_openloop(robot, dP, args)

            result.append({'percent': k*36/216, 'alpha': alpha[i], 'feedback': feedback})

    # clean feedback data
    for item in result:
        fb = item['feedback']
        fb['touch'] = fb['touch'][0][:, 1:3]
        fb['touch'][:,1] -= 300 
        fb['touch'] = la.norm(fb['touch'], axis=1)
        fb['pose'] = fb['pose'][0][:,[0,2]]

    with open(args.output_path, 'wb') as file:
        pickle.dump(result, file)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run robot closeloop simulation for 2000 times')
    parser.add_argument('--render', default=False, type = bool, help='render simulation')
    parser.add_argument('--break_thresh', default=0.06, type=float, help='velcro breaking threshold')
    parser.add_argument('--act_mag', default=2, type=float, help='robot action magnitude')
    parser.add_argument('--grip', default=300, type=int, help='grip force in each action excution')
    parser.add_argument('--max_iterations', default=1, type=int, help='grip force in each action excution')
    parser.add_argument('--num_tendon', default=216, type=int, help='total number of tendons')
    parser.add_argument('--hap_sample', default=150, type=int, help='number of haptics samples feedback in each action excution')
    parser.add_argument('--output_path', default='/home/jc/logs/tactile_visualize/test.pickle', help='file to store openloop test results')
    args = parser.parse_args()
    
    main(args)
