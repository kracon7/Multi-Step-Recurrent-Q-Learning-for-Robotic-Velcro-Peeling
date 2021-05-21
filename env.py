import os
import time
import math
import random
import pickle
import numpy as np
import matplotlib.pyplot as plt
from itertools import count
import datetime as dt

import torch
import torch.optim as optim
import torch.nn.functional as F
from mujoco_py import MjViewer, load_model_from_path, MjSim
from networks.a2c import ActorCritic
from robot_sim import RobotSim
from sim_param import SimParameter
from utils.action_buffer import ActionSpace, Observation
from utils.memory import RecurrentMemory, Transition
from utils.normalize import Normalizer
from utils.visualization import plot_variables
from utils.gripper_util import init_model

class RobotEnv:
    def __init__(self, args):
        self.args = args
        self.ACTIONS = ['left', 'right', 'forward', 'backward', 'up', 'down']  # 'open', 'close']
        self.gripping_force = args.grip_force
        self.break_threshold = args.break_thresh
        self.action_space = ActionSpace(dp=0.06, df=10)
        self.broken_so_far = 0
        self.robot = None
        self.model_params = None

    def reset(self):
        args = self.args
        model, self.model_params = init_model(args.model_path, geom=False)
        sim = MjSim(model)
        sim.step()
        if args.render:
            viewer = MjViewer(sim)
        else:
            viewer = None

        sim_param = SimParameter(sim)
        robot = RobotSim(sim, viewer, sim_param, args.render, self.break_threshold)
        self.robot = robot
        robot.reset_simulation()
        ret = robot.grasp_handle()
        if not ret:
            return None

        self.obs_space = Observation(robot.get_gripper_jpos(),
                                     robot.get_num_broken_buffer(args.hap_sample),
                                     robot.get_main_touch_buffer(args.hap_sample)) 

        # Get current observation
        self.broken_so_far = 0

        self.obs_space.update(robot.get_gripper_jpos(),            # 6
                                 robot.get_num_broken_buffer(args.hap_sample),         # 30x1
                                 robot.get_main_touch_buffer(args.hap_sample))     # 30x12

        obs = self.obs_space.get_state()

        reward = 0
        done = False
        info = {'slippage': False}
        return obs, reward, done, info

    def step(self, action):
        robot = self.robot
        args = self.args
        delta = self.action_space.get_action(self.ACTIONS[action])['delta'][:3]
        target_position = np.add(robot.get_gripper_jpos()[:3], np.array(delta))
        target_pose = np.hstack((target_position, robot.get_gripper_jpos()[3:]))
        robot.move_joint(target_pose, True, self.gripping_force, hap_sample = args.hap_sample)
        
        # Get reward
        done, num = robot.update_tendons()
        slippage = robot.check_slippage()
        if num > self.broken_so_far:
            reward = num - self.broken_so_far
            self.broken_so_far = num
        else:
            reward = 0

        if slippage:
            done = True

        info = {'slippage': slippage}

        self.obs_space.update(robot.get_gripper_jpos(),            # 6
                                 robot.get_num_broken_buffer(args.hap_sample),         # 30x2
                                 robot.get_main_touch_buffer(args.hap_sample))     # 30x7
        obs = self.obs_space.get_state()
        return obs, reward, done, info

