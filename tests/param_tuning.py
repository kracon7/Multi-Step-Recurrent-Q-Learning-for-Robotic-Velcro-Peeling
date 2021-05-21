import os
import sys
import argparse

# baseline demo for peel velcro with paraGripper
import numpy as np
import matplotlib.pyplot as plt
import math
from mujoco_py import MjViewer, load_model_from_path, MjSim

from robot_sim import RobotSim
from sim_param import SimParameter
from utils import gripper_util
from utils.action_buffer import get_action_sequence


# Helper function to plot duration of episodes and an average over the last 100 iterations
def plot_variables(fig, plot_var):
    fig, ax1, ax2, ax3 = fig
    pos, shear, touch = plot_var

    fig.suptitle('Robot Observations')
    ax1.set_xlabel('action steps')
    ax1.set_ylabel('pos')
    ax2.set_xlabel('action steps')
    ax2.set_ylabel('shear')
    ax3.set_xlabel('action steps')
    ax3.set_ylabel('touch')
    ax1.plot(pos)
    ax2.plot(shear)
    ax3.plot(touch)

    plt.pause(0.001)    # Small pause to update plots

def run_openloop(args, robot, ACTIONS, act_sequence, max_time, figs):
    result = {'done':False, 'time':max_time} 

    robot.reset_simulation()

    robot_pos = np.empty((0, 3))
    robot_shear = np.empty((0,8))
    robot_touch = np.empty((0,10))

    ######################################
    ## move gripper to grasp the handle ##
    ret = robot.grasp_handle()
    if ret:
        robot.checkTendons = True

        for t in range(max_time):
            for k in range(len(act_sequence)):
                dP = ACTIONS[act_sequence[k]]
                currentGripperJointValues = robot.get_gripper_jpos()
                jointValues_target = currentGripperJointValues + np.array([dP[0], dP[1], dP[2], 0, 0, 0])
                robot.move_joint(jointValues_target, fingerClose=True, fingerCloseForce=250)

                # get position shear and touch readings
                robot_pos = np.vstack((robot_pos, robot.get_gripper_jpos()[:3]))
                robot_shear = np.vstack((robot_shear, robot.read_shear_forces()))
                robot_touch = np.vstack((robot_touch, robot.read_touch_sensors()))

                # check tendons
                done, num_broken = robot.update_tendons()
                
                # check slippage
                slippage = robot.check_slippage()

                if done:
                    result['done'] = True 
                    result['time'] = t
                    if args.plot:
                        (fig,ax1,ax2,ax3) = figs
                        plot_variables((fig,ax1,ax2,ax3), (robot_pos, robot_shear, robot_touch))
                    return result
                if slippage:
                    return result
        return result
    else:
        return result


def main(args):
    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # test_xml = os.path.join(parent, 'tests/test_xmls/case1_flat_-0.6392_0.2779_0_0_2.6210_inf.xml')
    test_xml = os.path.join(parent, 'models/updateVelcro.xml')
    num_success = 0
    num_trial = 0
    total_time = 0
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ACTIONS = [0.06 * np.array([0, 0, 1]),
               0.06 * np.array([0, 1, 0]),
               0.06 * np.array([1, 0, 0]),
               0.06 * np.array([0, -1, 0]),
               0.06 * np.array([-1, 0, 0])]

    np.set_printoptions(suppress=True)


    # load model and sim parameters
    model = load_model_from_path(test_xml)
    sim = MjSim(model)
    if args.render:
        viewer = MjViewer(sim)
    else:
        viewer = None
    sim_param = SimParameter(sim)
    robot = RobotSim(sim, viewer, sim_param, args.render, 0.06)

    for i in range(args.num_try):
        num_trial = num_trial + 1
        # try n times for each test case 
        sequence = get_action_sequence()
        print('\n\nAction sequence is: {}'.format(sequence))
        result = run_openloop(args, robot, ACTIONS, sequence, 60, (fig, ax1, ax2, ax3))

        if result['done']:
            num_success = num_success + 1
            total_time = total_time + result['time']
            print('Execution succeed, it took {} steps, num trail is {}, success rate is {}'.format(
                        result['time'],num_trial, num_success / num_trial))


    success_rate = float(num_success) / float(args.num_try)
    if num_success > 0:
        ave_time = float(total_time) / float(num_success)
    else:
        ave_time = 0
    print('\n\nTotal number of trails is: {}, success rate is: {}%, average time it took is: {}'.format(
                num_trial, success_rate * 100, ave_time))
    print('=======================================================================================\n\n\n\n\n\n\n')


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run robot closeloop simulation for 2000 times')
    parser.add_argument('--render', default=False, type = bool, help='render simulation')
    parser.add_argument('--num_try', default=1, type = int, help='number of try per test case')    
    parser.add_argument('--plot', default=False, type=bool, help='turn on/off haptics plot')

    args = parser.parse_args()
    
    main(args)
