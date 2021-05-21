import os
import sys
import argparse

# baseline demo for peel velcro with paraGripper
import numpy as np
import matplotlib.pyplot as plt
import math
from mujoco_py import MjViewer, load_model_from_path, MjSim

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # /corl2019_learningHaptics
sys.path.insert(0, PARENT_DIR)
from robot_sim import RobotSim
from sim_param import SimParameter
from utils import gripper_util
from utils.velcro_utils import VelcroUtil
from utils.action_buffer import get_action_sequence


def run_action(args, robot, action):
    max_time = args.max_iterations
    result = {'done':False, 'time':max_time, 'num_broken': 0, 'tactile': np.empty((0, 6))} 

    robot.reset_simulation()
    ret = robot.grasp_handle()
    robot.checkTendons = True
    for t in range(max_time):
        currentGripperJointValues = robot.get_gripper_jpos()
        jointValues_target = currentGripperJointValues + np.array([action[0], action[1], action[2], 0, 0, 0])
        robot.move_joint(jointValues_target, fingerClose=True, fingerCloseForce=args.grip_force, hap_sample=args.hap_sample)

        result['tactile'] = np.vstack((result['tactile'], robot.feedback_buffer['main_touch']))

        # check tendons
        done, num_broken = robot.update_tendons()
        result['num_broken'] = num_broken     

        if done:
            result['done'] = True 
            result['time'] = t
            return result
    return result


def main(args):
    test_xml = os.path.join(PARENT_DIR, 'models/flat_velcro.xml')

    # load model and sim parameters
    model = load_model_from_path(test_xml)
    sim = MjSim(model)
    if args.render:
        viewer = MjViewer(sim)
    else:
        viewer = None
    sim_param = SimParameter(sim)
    robot = RobotSim(sim, viewer, sim_param, args.render, args.break_thresh)

    act_mag = args.act_mag
    act_z = act_mag * np.array([0, 0, 1])
    act_x = act_mag * np.array([1, 0, 0])

    np.set_printoptions(suppress=True)

    fig, axes = plt.subplots(5, 2)

    success = []
    N = args.num_increment
    for i in range(N):
        velcro_params = gripper_util.init_model(robot.mj_sim)

        theta = float(i + 1)/float(N) * 3.14159
        action = math.sin(theta) * act_z + math.cos(theta) * act_x
        result = run_action(args, robot, action)
        print(i, result['done'], result['num_broken'], sim.data.sensordata[:])
        if result['done']:
            success.append(1)
        else:
            success.append(0)

        axes[i//2, i%2].plot(result['tactile'][:,0], 'r')
        axes[i//2, i%2].plot(result['tactile'][:,1], 'g')
        axes[i//2, i%2].plot(result['tactile'][:,2], 'b')
        axes[i//2, i%2].set_ylim(-800, 800)
        plt.pause(0.01)
    # plt.savefig(os.path.join(PARENT_DIR, 'tests', '0.png'))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run robot closeloop simulation for 2000 times')
    parser.add_argument('--render', default=False, type = bool, help='render simulation')
    parser.add_argument('--case', default=1, help='case number')
    parser.add_argument('--num_try', default=1, type = int, help='number of try per test case') 
    parser.add_argument('--break_thresh', default=0.06, type=float, help='velcro breaking threshold')
    parser.add_argument('--act_mag', default=0.06, type=float, help='robot action magnitude')
    parser.add_argument('--grip_force', default=400, type=float, help='gripping force')
    parser.add_argument('--max_iterations', default=20, type=int, help='grip force in each action excution')
    parser.add_argument('--num_increment', default=10, type=int, help='number of discretized parts of half circle')
    parser.add_argument('--hap_sample', default=30, type=int, help='number of haptics sample frequency')
    parser.add_argument('--num_tendon', default=216, type=int, help='total number of tendons')
    parser.add_argument('--output_path', default='openloop_result.txt', help='file to store openloop test results')
    args = parser.parse_args()
    
    main(args)
