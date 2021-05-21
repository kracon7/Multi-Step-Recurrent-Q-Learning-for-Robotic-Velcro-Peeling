import os
import sys
import argparse

# baseline demo for peel velcro with paraGripper
import numpy as np
import matplotlib.pyplot as plt
import math
from mujoco_py import MjViewer, load_model_from_path, MjSim

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent)

from robot_sim import RobotSim
from sim_param import SimParameter
from utils import gripper_util

test_xml_names = []
test_xml_dir = os.path.join(parent, 'tests/test_xmls/')
for file in os.listdir(test_xml_dir):
    if 'case' in file:
        test_xml_names.append(file)

fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

np.set_printoptions(suppress=True)

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
    plt.show()
    # plt.pause(0.001)    # Small pause to update plots

def run_openloop(robot, dP, max_time):
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

        robot_shear = None
        robot_fpos = None
        robot_maintouch = None

        for k in range(max_time):
            currentGripperJointValues = robot.get_gripper_jpos()
            jointValues_target = currentGripperJointValues + np.array([dP[0], dP[1], dP[2], 0, 0, 0])

            robot.move_joint(jointValues_target, fingerClose=True, fingerCloseForce=200)
            robot_shear = robot.feedback_buffer['shear']
            robot_touch = robot.feedback_buffer['touch_array']
            robot_maintouch = robot.feedback_buffer['main_touch']
            # if robot_shear is None:
            #     robot_shear = robot.feedback_buffer['shear']
            #     robot_fpos = robot.feedback_buffer['fpos']
            #     robot_maintouch = robot.feedback_buffer['main_touch']
            # else:
            #     robot_shear = np.vstack((robot_shear, robot.feedback_buffer['shear']))
            #     robot_fpos = np.vstack((robot_fpos, robot.feedback_buffer['fpos']))
            #     robot_maintouch = np.vstack((robot_maintouch, robot.feedback_buffer['main_touch']))

            # check tendons
            done, num_broken = robot.update_tendons()
            
            # check slippage
            slippage = robot.check_slippage()

            if done:
                result['done'] = True 
                result['time'] = k
                # plot_variables((fig,ax1,ax2,ax3), (robot_fpos, robot_shear, robot_maintouch))
                return result
            if slippage:
                # plot_variables((fig,ax1,ax2,ax3), (robot_fpos, robot_shear, robot_maintouch))
                return result
        return result
    else:
        return result

def main(args):
    num_test = len(test_xml_names)
    num_success = 0
    num_trial = 0
    total_time = 0

    print('                    ++++++++++++++++++++++++++')
    print('                    +++ Now running case {} +++'.format(args.case))
    print('                    +++ Num try per case {} +++'.format(args.num_try))
    print('                    ++++++++++++++++++++++++++\n\n')

    for i in range(num_test):
        f = test_xml_names[i]
        if 'case{}'.format(args.case) in f:
            # one more trail for case1
            num_trial = num_trial + 1
            print('Now testing {}'.format(f))

            # load model and sim parameters
            model = load_model_from_path(os.path.join(test_xml_dir, f))
            sim = MjSim(model)
            if args.render:
                viewer = MjViewer(sim)
            else:
                viewer = None
            sim_param = SimParameter(sim)
            robot = RobotSim(sim, viewer, sim_param, args.render, 0.05)

            # try n times for each test case 
            for j in range(args.num_try):
                action_mag = 0.06
                alpha = np.random.randint(0, 4)
                beta = np.random.randint(0, 15)
                dP = np.array([math.sin(alpha * np.pi / 8) * math.cos(beta * np.pi / 6),
                               math.sin(alpha * np.pi / 8) * math.sin(beta * np.pi / 6),
                               math.cos(alpha * np.pi / 8)]) * action_mag
                result = run_openloop(robot, dP, 80)

                if result['done']:
                    num_success = num_success + 1
                    total_time = total_time + result['time']
                    print('Execution succeed, it took {} steps, num trail is {}, success rate is {}'.format(
                                result['time'],num_trial, num_success / num_trial))
                    break

    success_rate = float(num_success) / float(num_trial)
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
    parser.add_argument('--case', default=1, help='case number')
    parser.add_argument('--num_try', default=3, type = int, help='number of try per test case')    

    args = parser.parse_args()
    
    main(args)
