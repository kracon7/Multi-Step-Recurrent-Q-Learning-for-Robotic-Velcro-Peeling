import os
import sys
import argparse

# baseline demo for peel velcro with paraGripper
import numpy as np
import matplotlib.pyplot as plt
import math
from utils.gripper_util import *
from sim_param import SimParameter

from robot_sim import RobotSim

# model_path = '/home/jc/research/corl2019_learningHaptics/models/cylinderVelcro.xml'
# model = load_model_from_path(model_path)
# sim = MjSim(model)
# viewer = MjViewer(sim)
# sim_param = SimParameter(sim)

###################################### FLAT MODEL TEST
# # # get velcro nodes name and id
# # numX = 36
# # numY = 6
# # velcro_node_names=[]
# # velcro_node_bodyid=[]
# # velcro_jointid=[]
# # for i in range(numX):
# #     for j in range(numY):
# #         node_name = 'comp_{}_{}'.format(i, j)
# #         idx = sim.model._body_name2id[node_name]
# #         velcro_node_names.append(node_name)
# #         velcro_node_bodyid.append(idx)
# #         velcro_jointid.append(sim.model._joint_name2id[node_name])

# # euler = [0.4, 0.2,  1.5]
# # sim.model.body_quat[1] = rot_table(euler)
# # sim.model.body_quat[3] = rot_handle2(euler)

# # # for idx in velcro_node_bodyid:
# # #     pos = sim.model.body_pos[idx].copy()
# # #     sim.model.body_pos[idx] = rot_comp(euler, pos)
# # #     sim.model.body_ipos[idx] = rot_comp(euler, pos)

# # for idx in velcro_jointid:
# #     qposadr = sim.model.jnt_qposadr[idx]
# #     pos = sim.data.qpos[qposadr:qposadr+3].copy()
# #     sim.data.qpos[qposadr: qposadr+3] = rot_comp(euler, pos)


# # for i in range(5000000):
# #     sim.step()
# #     viewer.render()
#################################### END OF FLAT MODEL TEST

# sim_ref = MjSim(model)
# sim.data.qpos[:] = sim_ref.data.qpos[:].copy()
# rot_mag = np.array([np.pi / 6, np.pi / 6,  np.pi])
# euler = np.multiply(rot_mag, 2 * np.random.rand(3) -1)

# sim.model.body_quat[sim.model._body_name2id['table']] = rot_table(euler)
# sim.model.body_quat[sim.model._body_name2id['handle2']] = rot_handle2(euler)

# for idx in sim_param.velcro_node_jointid:
#     qposadr = sim.model.jnt_qposadr[idx]
#     pos = sim.data.qpos[qposadr:qposadr+3].copy()
#     sim.data.qpos[qposadr: qposadr+3] = rot_comp(euler, pos)

# for i in range(5000000):
#     sim.step()
#     viewer.render()


# sim_ref = MjSim(model)
# for i in range(5):
#     rot_model(sim_ref, sim, sim_param)
#     for j in range(10000):
#         sim.step()
#         viewer.render()





# ###################### TEST CHANGE MODEL RADIUS 
# sim_ref = MjSim(model)
# for i in range(3):
#     radius = 0.5 * (i+1)
#     change_model_radius(sim, sim_param, radius=radius)
#     sim.reset()
#     for j in range(10000):
#         sim.step()
#         viewer.render()
# ##################### END OF CHANGE MODEL RADIUS

model_path = '/home/jc/research/corl2019_learningHaptics/models/flat_velcro.xml'
model, sim, viewer= load_model(model_path)
sim_param = SimParameter(sim)

robot = RobotSim(sim, viewer, sim_param, True, 0.06)

for i in range(3):
    radius = 0.5 * (i+1)
    change_sim(sim, 'cylinder',[0.2, -0.5, 0], [0.2, 0, 1], radius)
    sim.reset()
    for j in range(2000):
        sim.step()
        viewer.render()

    change_sim(sim, 'flat',[0.2, -0.5, 0], [0, .2, 0.5])
    for j in range(2000):
        sim.step()
        viewer.render()

