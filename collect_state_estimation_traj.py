import os
import sys
import argparse
import random
import pickle
from itertools import count
import numpy as np
from statistics import mean

import torch
from mujoco_py import MjViewer, load_model_from_path, MjSim

from networks.dqn import Geom_DQN
from robot_sim import RobotSim
from sim_param import SimParameter
import matplotlib.pyplot as plt

from utils.normalize import Normalizer, Multimodal_Normalizer, Geom_Normalizer
from utils.action_buffer import ActionSpace, Observation, TactileObs
from utils.velcro_utils import VelcroUtil
from utils.gripper_util import init_model, init_for_test, norm_img, norm_depth

plt.ion()

def write_results(path, results):
	f = open(path, 'wb')
	pickle.dump(results, f)
	f.close()

class TrajLogger:
	def __init__(self, args, robot, velcro_util, tactile_obs_space):
		self.args = args
		self.ACTIONS = ['left', 'right', 'forward', 'backward', 'up', 'down']
		self.action_space = ActionSpace(dp=2.5*args.act_mag, df=10)
		self.robot = robot
		self.velcro_util = velcro_util
		self.tactile_obs_space = tactile_obs_space
		self.normalizer = Multimodal_Normalizer(num_inputs = args.indim, device=args.device)
		self.normalizer.restore_state(args.normalizer_file)

	def select_action(self):
		sample = random.random()
		p_threshold = self.args.p
		if sample > p_threshold:
			return self.expert_action()
		else:
			action = random.randrange(6)
			return self.action_space.get_action(self.ACTIONS[action])['delta'][:3]

	def expert_action(self):
		norms = self.velcro_util.break_norm()
		centers = self.velcro_util.break_center()
		fl_center = centers[:3]
		fs_center = centers[3:]
		fl_norm = norms[:3]
		fs_norm = norms[3:6]
		break_dir_norm = norms[6:9]

		action_direction = self.args.act_mag*(-0.5 * fl_norm + 0.5 * break_dir_norm)
		return action_direction
			
	def test_network(self, performance):
		args = self.args
		max_iterations = args.max_iter

		broken_so_far = 0
		expert_traj = []
		
		for t in range(max_iterations):
			delta = self.select_action()

			# sample tactile feedback and norm info 
			tactile_obs = self.tactile_obs_space.get_state()
			tactile_obs = self.normalizer.normalize(tactile_obs)

			expert_traj.append({'tactile': tactile_obs[:366], 'norm': self.velcro_util.break_norm(), 
						'center': self.velcro_util.break_center(), 'gpos': self.robot.get_gripper_jpos()[:3]})

			# perform action
			target_position = np.add(self.robot.get_gripper_jpos()[:3], np.array(delta))
			target_pose = np.hstack((target_position, self.robot.get_gripper_jpos()[3:]))
			self.robot.move_joint(target_pose, True, args.grip_force, hap_sample = args.hap_sample)
			
			self.tactile_obs_space.update(self.robot.get_gripper_jpos(),            # 6
                                             self.robot.get_all_touch_buffer(args.hap_sample))     # 30x12

			# Get reward
			done, num = self.robot.update_tendons()
			failure = self.robot.check_slippage()
			if num > broken_so_far:
				broken_so_far = num
  
			if done:
				performance['success'].append(1)
				performance['time'].append(t + 1)
				return performance, expert_traj
				break
			if failure:
				performance['success'].append(0)
				performance['time'].append(t + 1)
				return performance, expert_traj
				break
		# exceed max iterations
		performance['success'].append(0)
		performance['time'].append(max_iterations)
		return performance, expert_traj

def main(args):

	if not os.path.isdir(os.path.dirname(args.result_path)):
		os.makedirs(os.path.dirname(args.result_path))

	# Create robot, reset simulation and grasp handle
	model = load_model_from_path(args.model_path)
	sim = MjSim(model)
	sim_param = SimParameter(sim)
	sim.step()
	if args.render:
		viewer = MjViewer(sim)
	else:
		viewer = None

	robot = RobotSim(sim, viewer, sim_param, args.render, args.break_thresh)
	velcro_util = VelcroUtil(robot, robot.mj_sim_param)
	tactile_obs_space = TactileObs(robot.get_gripper_jpos(),            # 6
							 robot.get_all_touch_buffer(args.hap_sample))   # 30 x 12

	traj_logger = TrajLogger(args, robot, velcro_util, tactile_obs_space)

	all_trajectories = []
	performance = {'time':[], 'success':[], 'action_hist':[0,0,0,0,0,0]}

	for i in range(args.num_traj):
		velcro_param = init_model(robot.mj_sim)
		robot.reset_simulation()
		ret = robot.grasp_handle()
		performance, expert_traj = traj_logger.test_network(performance)
		if len(expert_traj)>0:
			all_trajectories.append(expert_traj)

		print('\n\nFinished trajectory {}, sampled {} steps in this episode'.format(i, len(expert_traj)))
		geom_type, origin_offset, euler, radius = velcro_param
		print('Velcro parameters are:{} {} {} {}'.format(geom_type, origin_offset, euler, radius))
		print(performance)

	print('\nCollected {} successful expert trajectories in total'.format(len(all_trajectories)))
	output = {'args': args, 'traj': all_trajectories}
	write_results(args.result_path, output)


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='PyTorch Tactile Test')
	parser.add_argument('--model_path', required=True, help='XML model to load')
	parser.add_argument('--device', default='cuda', help='device')
	parser.add_argument('--break_thresh', default=0.06, type=float, help='velcro breaking threshold')
	parser.add_argument('--indim', default=396, type=int, help='observation space size')
	parser.add_argument('--act_mag', default=0.1, type=float, help='robot action magnitude')
	parser.add_argument('--max_iter', default=150, type=int, help='max number of iterations per epoch')
	parser.add_argument('--grip_force', default=300, type=float, help='gripping force')
	parser.add_argument('--result_path', default='.', help='path where to save')
	parser.add_argument('--quiet', action='store_true', help='wether to print episodes or not')
	parser.add_argument('--render', default=False, type=bool, help='turn on rendering')
	parser.add_argument('--num_traj', default=50, type=int, help='case number')
	parser.add_argument('--p', default=0.6, type=float, help='randomness threshold for action selection')
	parser.add_argument('--sample_ratio', default=0.1, type=float, help='randomness threshold for sample data')
	parser.add_argument('--img_w', default=250, type=int, help='observation image width')
	parser.add_argument('--img_h', default=250, type=int, help='observation image height')
	parser.add_argument('--depth', default=True, type=bool, help='use depth from rendering as input')
	parser.add_argument('--hap_sample', default=30, type=int, help='number of haptics samples feedback in each action excution')
	parser.add_argument('--normalizer_file', default='/home/jc/logs/tactile_normalizer.pickle', help='normalizer file to load to resume training')
	
	args = parser.parse_args()
	main(args)
