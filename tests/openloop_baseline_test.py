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
from utils.velcro_utils import VelcroUtil
from utils.action_buffer import get_action_sequence


def run_openloop(args, robot, ACTIONS, act_sequence, velcro_util):
    max_time = args.max_iterations * 6
    result = {'done':False, 'time':max_time, 'num_broken': 0} 

    robot.reset_simulation()

    ######################################
    ## move gripper to grasp the handle ##
    ret = robot.grasp_handle()
    if ret:
        robot.checkTendons = True
        max_time = args.max_iterations
        for t in range(max_time):
            for k in range(len(act_sequence)):
                dP = ACTIONS[act_sequence[k]]
                currentGripperJointValues = robot.get_gripper_jpos()
                jointValues_target = currentGripperJointValues + np.array([dP[0], dP[1], dP[2], 0, 0, 0])
                robot.move_joint(jointValues_target, fingerClose=True, fingerCloseForce=args.grip)

                # check tendons
                done, num_broken = robot.update_tendons()
                result['num_broken'] = num_broken
                
                # check slippage
                slippage = robot.check_slippage()

                if done:
                    result['done'] = True 
                    result['time'] = t * len(act_sequence) + k
                    return result
                if slippage:
                    return result
        return result
    else:
        return result


def main(args):
    parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, parent)
    test_xml_names = []
    test_xml_dir = os.path.join(parent, 'tests/test_xmls/')
    for file in os.listdir(test_xml_dir):
        if 'case' in file:
            test_xml_names.append(file)

    act_mag = args.act_mag
    ACTIONS = [act_mag * np.array([0, 0, 1]),
               act_mag * np.array([0, 1, 0]),
               act_mag * np.array([1, 0, 0]),
               act_mag * np.array([0, -1, 0]),
               act_mag * np.array([-1, 0, 0])]

    np.set_printoptions(suppress=True)

    num_test = len(test_xml_names)
    num_success = 0
    num_trial = 0
    tendon_histogram = [0, 0, 0, 0]
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
            # print('Now testing {}'.format(f))

            # load model and sim parameters
            model = load_model_from_path(os.path.join(test_xml_dir, f))
            sim = MjSim(model)
            if args.render:
                viewer = MjViewer(sim)
            else:
                viewer = None
            sim_param = SimParameter(sim)
            robot = RobotSim(sim, viewer, sim_param, args.render, args.break_thresh)
            velcro_util = VelcroUtil(robot, sim_param)

            # try n times for each test case 
            for j in range(args.num_try):
                sequence = get_action_sequence()
                print('\n\nAction sequence is: {}'.format(sequence))
                result = run_openloop(args, robot, ACTIONS, sequence, velcro_util)
                print('Total number of broken tendon is: {}'.format(result['num_broken']))
                ratio_broken = float(result['num_broken']) / float(args.num_tendon)
                if ratio_broken < 0.25:
                    tendon_histogram[0] += 1
                elif ratio_broken >= 0.25 and ratio_broken < 0.5:
                    tendon_histogram[1] += 1
                elif ratio_broken >= 0.5 and ratio_broken < 0.75:
                    tendon_histogram[2] += 1
                else:
                    tendon_histogram[3] += 1

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

    with open(args.output_path, 'w') as f:
        f.write('Total number of trails is: {}, success rate is: {}%, average time it took is: {}\n'.format(
                num_trial, success_rate * 100, ave_time))
        f.write('Tendon histogram is {}'.format(tendon_histogram))
    f.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run robot closeloop simulation for 2000 times')
    parser.add_argument('--render', default=False, type = bool, help='render simulation')
    parser.add_argument('--case', default=1, help='case number')
    parser.add_argument('--num_try', default=1, type = int, help='number of try per test case') 
    parser.add_argument('--break_thresh', default=0.06, type=float, help='velcro breaking threshold')
    parser.add_argument('--act_mag', default=0.06, type=float, help='robot action magnitude')
    parser.add_argument('--grip', default=300, type=int, help='grip force in each action excution')
    parser.add_argument('--max_iterations', default=30, type=int, help='grip force in each action excution')
    parser.add_argument('--num_tendon', default=216, type=int, help='total number of tendons')
    parser.add_argument('--output_path', default='openloop_result.txt', help='file to store openloop test results')
    args = parser.parse_args()
    
    main(args)
