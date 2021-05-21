import numpy as np
import numpy.linalg as la

class VelcroUtil():
	def __init__(self, Robot, SimParam):
		self.robot = Robot
		self.sim_param = SimParam
		self.front_line_prev = None
		self.front_surface_prev = None
		self.break_direction_norm_prev = None

	def break_norm(self):
		## tendon_binary is True is tendons are still attached
		nodes_xpos = self.robot.get_table_node_xpos()
		tendon_binary = self.robot.get_tendon_binary()
		handle_xpos = self.robot.mj_sim.data.body_xpos[self.robot.mj_sim.model._body_name2id['gripperLink_0']]

		tendon_names = self.sim_param.velcro_tendon_names
		if np.sum(tendon_binary) > 12:
			front_line = nodes_xpos[tendon_binary.astype(bool)][:12]
			self.front_line_prev = front_line
		else:
			front_line = self.front_line_prev
		front_line_norm = self.estimate_norm(front_line)

		if np.sum(tendon_binary) > 36:
			front_surface = nodes_xpos[tendon_binary.astype(bool)][24:36]
			self.front_surface_prev = front_surface
			break_direction = np.average(front_surface, axis=0) - np.average(front_line, axis=0)
			break_direction_norm = break_direction / la.norm(break_direction)
			self.break_direction_norm_prev = break_direction_norm
		else:
			front_surface = self.front_surface_prev
			break_direction_norm = self.break_direction_norm_prev
		front_surface_norm = self.estimate_norm(front_surface)

		handle_direction = handle_xpos - np.average(front_line, axis=0)
		handle_direction_norm = handle_direction / la.norm(handle_direction)

		rotz = self.robot.get_robot_rotz()
		
		return np.concatenate((rotz @ front_line_norm, rotz @ front_surface_norm, rotz @ break_direction_norm, rotz @ handle_direction_norm))

	def break_center(self):
		## tendon_binary is True is tendons are still attached
		nodes_xpos = self.robot.get_table_node_xpos()
		tendon_binary = self.robot.get_tendon_binary()
		tendon_names = self.sim_param.velcro_tendon_names
		if np.sum(tendon_binary) > 12:
			front_line = nodes_xpos[tendon_binary.astype(bool)][:12]
			self.front_line_prev = front_line
		else:
			front_line = self.front_line_prev

		if np.sum(tendon_binary) > 36:
			front_surface = nodes_xpos[tendon_binary.astype(bool)][24:36]
			self.front_surface_prev = front_surface
		else:
			front_surface = self.front_surface_prev
		rotz = self.robot.get_robot_rotz()
		return np.concatenate((rotz @ np.average(front_line, axis=0), rotz @ np.average(front_surface, axis=0) ))


	def estimate_norm(self, xpos):
		center = np.average(xpos,axis=0)
		vec = xpos - center
		U = la.svd(vec.T@vec)[0]
		norm = U[:,-1]

		## always make norm pointing outwards
		center_vec = self.get_center_vec(xpos)
		if norm @ center_vec < 0:
			norm = -norm
		return norm

	def get_center_vec(self, xpos):
		center = np.average(xpos, axis=0)
		table_id = self.robot.mj_sim.model._geom_name2id["tabletop"]
		table_xpos = self.robot.mj_sim.data.body_xpos[table_id]
		return center - table_xpos