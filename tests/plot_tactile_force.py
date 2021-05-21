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

def main(args):
    num_tests = 4
    N = 10
    with open('/home/jc/logs/tactile_visualize/test.pickle', 'rb') as file:
        result = pickle.load(file)


    fig, axe = plt.subplots(1, 1)
    for k in range(num_tests):
        ret = result[k*(N-1) : (k+1)*(N-1)]
        for item in ret:
            fb = item['feedback']
            fb['touch'][fb['touch'] > 500] = 500
            for i in range(args.hap_sample): 
                color = fb['touch'][i] / 500
                axe.plot(fb['pose'][i, 0], fb['pose'][i, 1], marker='.', color=(1, 0, 0, color))
        fig.savefig('num{}'.format(k), format='png')
        axe.cla()

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
