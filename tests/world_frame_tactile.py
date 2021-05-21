import os
import sys
import argparse

# baseline demo for peel velcro with paraGripper
import numpy as np
import matplotlib.pyplot as plt
import math

PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PARENT_DIR)
from utils.gripper_util import *
from sim_param import SimParameter

from robot_sim import RobotSim


model_path = os.path.join(PARENT_DIR, 'models/flat_velcro.xml')
model, sim, viewer= load_model(model_path)
sim_param = SimParameter(sim)
robot = RobotSim(sim, viewer, sim_param, True, 0.06)
np.set_printoptions(suppress=True)

for i in range(3):
    change_sim(sim, 'flat',[0.2, -0.5, 0], [0, 0.3, 0.5])
    robot.reset_simulation()
    robot.grasp_handle()
    
    sim.data.ctrl[sim_param.gripper_pos_ctrl_id] = np.array([0, 0, 0.5, 0, 0, 0])
    sim.data.ctrl[sim_param.gripper_vel_ctrl_id] = np.array([0, 0, 0.5, 0, 0, 0])
    for i in range(1000):
        robot.mj_simulation_step()
        viewer.render()

    for j in range(20000):
        viewer.render()

    radius = 0.5 * (i+1)
    change_sim(sim, 'cylinder',[0.2, -0.5, 0], [1, 0.7, 1], radius)
    sim.reset()
    for j in range(20000):
        sim.step()
        viewer.render()

