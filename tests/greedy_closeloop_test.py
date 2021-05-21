import os
import sys
import argparse

# baseline demo for peel velcro with paraGripper
import numpy as np
import numpy.linalg as la
import scipy.special as spec
from mujoco_py import MjViewer, load_model_from_path, MjSim

parent = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent)

from robot_sim import RobotSim
from sim_param import SimParameter

test_xml_names = []
test_xml_dir = os.path.join(parent, 'tests/test_xmls/')
for file in os.listdir(test_xml_dir):
    if 'case' in file:
        test_xml_names.append(file)

np.set_printoptions(suppress=True)

def choose_action(eta, k = 1.4):
    eta = spec.softmax(eta)
    if np.amax(eta) > k / 6:
        return eta.argmax()
    else:
        return np.random.randint(6)

def run_closeloop(robot, action_mag, actions, reward_discount = 2, max_time = 300):
    eta = np.ones(6)
    num_act = np.ones(6)
    num_broken_prev = 0
    x = np.random.randint(6)

    result = {'done':False, 'time':max_time} 

    robot.reset_simulation()

    ######################################
    ## move gripper to grasp the handle ##
    ret = robot.grasp_handle()
    if ret:
        robot.checkTendons = True

        for t in range(max_time):
            action = actions[x]    # size 6
            jpos_current = robot.get_gripper_jpos()
            jpos_target = jpos_current + action
            p_current = jpos_current[0:3]
            p_target = jpos_target[0:3]

            robot.move_joint(jpos_target, fingerClose=True, fingerCloseForce=120)
            jpos_prev = jpos_current
            p_prev = jpos_prev[0:3]
            jpos_current = robot.get_gripper_jpos()
            p_current = jpos_current[0:3]

            # evaluate execution
            motion_eval = la.norm(p_current - p_prev) / action_mag
            # check tendons
            done, num_broken = robot.update_tendons()
            tendon_rewards = (num_broken - num_broken_prev)/reward_discount
            # check slippage
            slippage = robot.check_slippage()
            # update action score
            eta[x] = (eta[x] * num_act[x] + motion_eval + tendon_rewards) / (num_act[x] + 1)
            
            num_broken_prev = num_broken
            num_act[x] = num_act[x] + 1

            # pick optimal actions from softmax(eta)
            # if max one is smaller than 2/num(actions), to random exploration
            x = choose_action(eta)

            if done:
                result['done'] = True 
                result['time'] = t
                return result
                break
            if slippage:
                return result
                break
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
            robot = RobotSim(sim, viewer, sim_param, args.render, 0.05)

            # try n times for each test case 
            action_mag = 0.05
            actions = [action_mag * np.array([1, 0, 0, 0, 0, 0]),
                       action_mag * np.array([-1, 0, 0, 0, 0, 0]),
                       action_mag * np.array([0, 1, 0, 0, 0, 0]),
                       action_mag * np.array([0, -1, 0, 0, 0, 0]),
                       action_mag * np.array([0, 0, 1, 0, 0, 0]),
                       action_mag * np.array([0, 0, -1, 0, 0, 0]), ]

            result = run_closeloop(robot, action_mag, actions)

            if result['done']:
                num_success = num_success + 1
                total_time = total_time + result['time']
                print('Execution succeed, it took {} steps, num trail is {}, success rate is {}'.format(
                            result['time'],num_trial, num_success / num_trial))


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
    parser.add_argument('--num_try', default=1, type = int, help='number of try per test case')    

    args = parser.parse_args()
    
    main(args)


