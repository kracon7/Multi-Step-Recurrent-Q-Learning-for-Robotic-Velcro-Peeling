#########################
#  Para Gripper class
########################

import numpy as np
import numpy.linalg as la
import math
import copy
from mujoco_py import MjViewer

from utils import gripper_util


class RobotSim():
    def __init__(self, mujoco_sim, mujoco_viewer, sim_param, render=False, velcro_break_thresh=None):
        # When initializing, extract qpos id and ctrl id
        # qpos id for check joint values and velocities
        # ctrl id for assigning commands to position/velocity actuators
        self.mj_sim = mujoco_sim
        self.mj_viewer = mujoco_viewer
        self.mj_sim_param = sim_param
        self.render = render
        self.velcro_break_thresh = velcro_break_thresh
        self.checkTendons = True
        self.feedback_buffer = {"main_touch": None, "fpos": None,  "num_broken": None, "tool_pose": None, "collision": 0}
        self.base_quat = copy.deepcopy(mujoco_sim.data.body_xquat[mujoco_sim.model._body_name2id['gripperLink_0']])
        self.num_broken = 0

    def move_joint(self, jointValues_target, fingerClose=True, fingerCloseForce=300, 
                    fingerOpenForce=1, hap_sample=30):
        # fingerClose controls finger open/close
        # n_hap_sample controls haptics history sample frequency, 0 turn off hist log and only take haptics reading from the last timestep
        currentGripperJointValues = self.get_gripper_jpos()
        g_qpos_id = self.mj_sim_param.gripper_qpos_id
        f_qpos_id = self.mj_sim_param.finger_qpos_id
        g_pos_ctrl_id = self.mj_sim_param.gripper_pos_ctrl_id
        g_vel_ctrl_id = self.mj_sim_param.gripper_vel_ctrl_id
        ret, traj = gripper_util.p2p_trajectory(currentGripperJointValues, jointValues_target, resolution=0.00025)

        if not ret:
            print('Goal is very close to start point, no need to move end effector.\n')
        else:
            vel_command = gripper_util.get_vel_command(traj, self.mj_sim_param.timeStep)
            pos_command = traj[1:]
            nStep = len(vel_command)

            if hap_sample > 0:  
                # if haptics sampling is turned on, use robot.feedback_buffer to achieve unsynchronized data collection
                sample_interval = int((nStep - (nStep % hap_sample))/hap_sample)
                # init empty ndarray for haptics feedback
                main_torch_log = [np.zeros(6) for i in range(5)]
                main_touch_buffer = np.empty((0,6))
                fpos_buffer = np.empty((0,2))
                num_broken_buffer = np.empty((0,1))
                tool_pose_buffer = np.empty((0,6))
                collision = 0

            for t in range(nStep):
                # gravity compensation for arm joints
                self.mj_sim.data.qfrc_applied[g_qpos_id] = self.mj_sim.data.qfrc_bias[g_qpos_id]
                # assign velocity and position actuator commands
                self.mj_sim.data.ctrl[g_pos_ctrl_id] = pos_command[t]
                self.mj_sim.data.ctrl[g_vel_ctrl_id] = vel_command[t]

                # assign finger actuator commands to remain in home config, i.e. position = velocity = 0
                if fingerClose:
                    self.mj_sim.data.qfrc_applied[f_qpos_id] = self.mj_sim.data.qfrc_bias[f_qpos_id] + fingerCloseForce
                else:
                    self.mj_sim.data.qfrc_applied[f_qpos_id] = self.mj_sim.data.qfrc_bias[f_qpos_id] - fingerOpenForce

                self.mj_simulation_step()
                tactile_force = self.read_main_touch() + np.array([0, 0, -fingerCloseForce, 0, 0, -fingerCloseForce])
                # main_torch_log.append(tactile_force)
                main_torch_log.append(self.tactile_in_world(tactile_force))

                if hap_sample > 0 and t < nStep - (nStep % hap_sample)  and t % sample_interval == 0:
                    tactile_ave = np.average(np.vstack(main_torch_log[-5:]), axis=0)
                    main_touch_buffer = np.vstack((main_touch_buffer, tactile_ave))
                    fpos_buffer = np.vstack((fpos_buffer, self.get_finger_jpos()))
                    tool_pose_buffer = np.vstack((tool_pose_buffer, self.get_gripper_jpos()))
                    
                    num_broken_so_far = np.sum(1 - self.get_tendon_binary())
                    num_broken_buffer = np.vstack((num_broken_buffer, num_broken_so_far - self.num_broken))
                    self.num_broken = num_broken_so_far
                if self.handle_table_collision():
                    collision += 1
                        
            if hap_sample > 0:
                # store feedback data in robot.feedback_buffer
                self.store_fbbuffer(main_touch_buffer, fpos_buffer, num_broken_buffer, tool_pose_buffer)
                self.feedback_buffer['collision'] = collision
                
            # get EE pose after moving gripper
            ee_pose = self.get_ee_pose()
            jointValues = gripper_util.ee_pose2jpos(ee_pose)

            # check error
            err = la.norm(self.get_gripper_jpos() - jointValues)
            if err < 5e-3:
                # print('Moving toolPose succedd!\nNow end effector at position: {}\n\n'.format(ee_pose))
                return True
            else:
                # print('Failed to reach goal state\nError is {}, Now end effector at position: {}'.format(err, ee_pose))
                return False

    def tactile_in_world(self, force):
        '''
        transform tactile force into world frame
        Input: 
            force -- left finger force and right finger force
        '''
        lxmat = self.mj_sim.data.body_xmat[self.mj_sim.model._body_name2id['left_finger_skin']].reshape(3,3)
        left_transformed = lxmat @ force[:3]
        rxmat = self.mj_sim.data.body_xmat[self.mj_sim.model._body_name2id['right_finger_skin']].reshape(3,3)
        right_transformed = rxmat @ force[3:]
        return np.concatenate([left_transformed, right_transformed])

    def apply_joint_force(self, force):
        g_qpos_id = self.mj_sim_param.gripper_qpos_id
        self.mj_sim.data.qfrc_applied[g_qpos_id] = np.array(force)

    def robot_freeze(self, numStep=1000):
        # print('===========================================================================')
        # freeze the robot for numStep steps, helpful during debug
        currentGripperJointValues = self.get_gripper_jpos()
        currentFingerJointValues = self.get_finger_jpos()

        # get joint and actuator id for control
        g_qpos_id = self.mj_sim_param.gripper_qpos_id
        f_qpos_id = self.mj_sim_param.finger_qpos_id
        g_pos_ctrl_id = self.mj_sim_param.gripper_pos_ctrl_id
        g_vel_ctrl_id = self.mj_sim_param.gripper_vel_ctrl_id
        f_pos_ctrl_id = self.mj_sim_param.finger_pos_ctrl_id
        f_vel_ctrl_id = self.mj_sim_param.finger_vel_ctrl_id
        # print('Robot in freeze mode, arm is in {}, gripper in home position\n'.format(currentGripperJointValues))

        for t in range(numStep):
            # gravity compensation
            self.mj_sim.data.qfrc_applied[g_qpos_id] = self.mj_sim.data.qfrc_bias[g_qpos_id]
            self.mj_sim.data.qfrc_applied[f_qpos_id] = self.mj_sim.data.qfrc_bias[f_qpos_id]
            # assign velocity and position actuator commands for gripper
            self.mj_sim.data.ctrl[g_pos_ctrl_id] = currentGripperJointValues
            self.mj_sim.data.ctrl[g_vel_ctrl_id] = np.zeros(currentGripperJointValues.shape)
            # assign velocity and position actuator commands for finger
            self.mj_sim.data.ctrl[f_pos_ctrl_id] = currentFingerJointValues[0]
            self.mj_sim.data.ctrl[f_vel_ctrl_id] = 0
            self.mj_simulation_step()

    def get_img(self, w, h, camera_name, depth=False):
        if self.mj_viewer is None:
            self.mj_viewer = MjViewer(self.mj_sim)
        img = self.mj_sim.render(width=w, height=h, camera_name=camera_name, depth=depth)
        img = self.apply_noise(w, h, img, depth)
        return img

    def handle_table_collision(self):
        rtip_id = self.mj_sim.model._geom_name2id['right_tip']
        ltip_id = self.mj_sim.model._geom_name2id['left_tip']
        handle_geomid = self.mj_sim.model._geom_name2id['handle']
        handle2_geomid = self.mj_sim.model._geom_name2id['handle2']
        table_geomid = self.mj_sim.model._geom_name2id['tabletop']
        result = False
        result = result or self.collision_check(ltip_id, handle_geomid)
        result = result or self.collision_check(rtip_id, handle_geomid)
        # result = result or self.collision_check(ltip_id, handle2_geomid)
        # result = result or self.collision_check(rtip_id, handle2_geomid)
        result = result or self.collision_check(ltip_id, table_geomid)
        result = result or self.collision_check(rtip_id, table_geomid)
        result = result or self.collision_check(table_geomid, handle_geomid)
        return result

    def collision_check(self, g1, g2):
        ncon = self.mj_sim.data.ncon
        collision = False
        if ncon > 0:
            for i in range(ncon):
                geom1 = self.mj_sim.data.contact[i].geom1
                geom2 = self.mj_sim.data.contact[i].geom2
                if (geom1 == g1 and geom2 == g2) or (geom2 == g1 and geom1 == g2):
                    collision = True
                    break
        return collision

    def apply_noise(self, w, h, img, depth):
        if depth:
            rgb = img[0]
            depth = img[1]
            rgb += np.random.normal(0, 2, (w,h,3)).astype('uint8')
            rgb[rgb < 0] = 0
            alpha = 0.02
            for i in range(w):
                for j in range(h):
                    depth[i,j] = np.random.normal(0, alpha*depth[i,j]) + depth[i,j]
            depth[depth<0] = 0
            img = (rgb, depth)
        else:
            img += np.random.normal(0, 2, (w,h,3))
            img[img < 0] = 0
        return img


    def mj_simulation_step(self):
        if self.checkTendons:
            done, num_broken = self.update_tendons()   # done is true when all tendons are broken
        else:
            done = False
            num_broken = 0

        self.mj_sim.step()
        if self.render:
            self.mj_viewer.render()

    def reset_simulation(self):
        self.mj_sim.reset()
        self.reset_tendons()

        #clear touch buffer
        for key in self.feedback_buffer:
            self.feedback_buffer[key] = None
            
        # after reset tendons and sim, make this False before any sim.step()...and do robot.robot_freeze()
        # for a few hunderds of timesteps to allow simulation reset stablize itself
        self.robot_freeze(numStep=500)
        return True

    def reset_tendons(self):
        # assign corresponding element in tendon_stiff with tendon_names to reset the model
        model = self.mj_sim.model
        tendon_names = self.mj_sim_param.velcro_tendon_names          # list of names
        stiff_init = self.mj_sim_param.velcro_tendon_stiff_init       # ndarray of stiffness for reset model
        color_init = self.mj_sim_param.velcro_tendon_color_init       # ndarray of color for reset model
        assert tendon_names is not None, 'Velcro tendons need initialization.\n'
        assert len(tendon_names) == stiff_init.shape[0], 'Name list must have the same length as stiffness list.\n'
        assert len(tendon_names) == color_init.shape[0], 'Name list must have the same length as color list.\n'
        for i, name in enumerate(tendon_names):
            tendon_id = model._tendon_name2id[name]
            model.tendon_stiffness[tendon_id] = stiff_init[i]
            model.tendon_rgba[tendon_id] = color_init[i]

    def update_tendons(self):
        num = 0         # number of broken tendons
        done = False    # whether or not all tendons are broken
        sim = self.mj_sim
        idx = self.mj_sim_param.velcro_tendon_id
        tendon_names = self.mj_sim_param.velcro_tendon_names
        break_thresh = self.velcro_break_thresh
        for name in tendon_names:
            tendon_id = sim.model._tendon_name2id[name]
            tendonLength = sim.data.ten_length[tendon_id]
            if sim.model.tendon_stiffness[tendon_id] == 0:
                num = num + 1
            elif sim.model.tendon_stiffness[tendon_id] > 0 and tendonLength > break_thresh:
                # set tendon stiffness to zero and make it transparent
                sim.model.tendon_stiffness[tendon_id] = 0
                sim.model.tendon_rgba[tendon_id] = np.array([0, 0, 0, 0])
                num = num + 1
        if num is len(tendon_names):
            done = True
        return done, num

    def check_slippage(self):
        handle_offset = 0.045
        finger1_pos, finger2_pos = self.get_finger_jpos()
        eps = 0.01

        if (handle_offset - finger1_pos) + (handle_offset - finger2_pos) < eps:
            return True

        return False

    def grasp_handle(self, force = 120, direction='z'):
        if direction == 'z':   # approaching from z direction
            handle2_id = self.mj_sim.model._geom_name2id['handle2']
            gripper_id = self.mj_sim.model._body_name2id['gripperLink_0']
            # compute the joint values which the gripper will be if grasp is successful
            joint_target = np.zeros(6)
            rotz = self.get_robot_rotz()
            d = self.mj_sim.data.geom_xpos[handle2_id] - self.mj_sim.data.body_xpos[gripper_id]
            joint_target[0:3] = np.dot(rotz, d)
            joint_target[3:] = np.array([-1.57, 0, 0])
            g_qpos_id = self.mj_sim_param.gripper_qpos_id
            self.mj_sim.data.qpos[g_qpos_id] = joint_target
            self.mj_sim.step()          # step simulation for one step to set joint angles
        ret = self.close_gripper(force=force)
        return ret

    def get_robot_rotz(self):
        q = self.base_quat
        s_z2 = q[3]
        c_z2 = q[0]
        s_z = 2 * s_z2 * c_z2
        c_z = c_z2**2 - s_z2**2
        return np.array([[c_z, s_z, 0],
                         [-s_z, c_z,  0],
                         [0,   0,    1]])

    def reorient_robot(self, rpy = None):
        if rpy is None: # if rpy is not given, randomly re-orient
            # x-y-z, x and y between -pi/4 to pi/4, z between -pi/2 to pi/2
            rpy = np.multiply(np.array([np.pi/4, np.pi/4, np.pi/2]), np.random.rand(3) - np.array([0.5, 0.5, 0]))
        else:   # otherwise clip roll pitch yaw angles
            rpy_min = np.array([-np.pi/4, -np.pi/4, -np.pi/2])
            rpy_max = np.array([np.pi/4, np.pi/4, np.pi/2])
            rpy = np.clip(rpy, rpy_min, rpy_max)
        R = gripper_util.roll_pitch_yaw(rpy)
        quat = gripper_util.rotation2Quaternion(R)   # use rotation matrix in world frame
        body_id = self.mj_sim.model._body_name2id['gripperLink_0']
        self.mj_sim.model.body_quat[body_id] = quat
        # print('Robot frame reset successfully, frame euler angle is {} wrt world frame.'.format(rpy))


    def close_gripper(self, force=10, maxTime = .15):
        # maintain arm joint configurations while moving gripper joints
        # print('===========================================================================')
        # initialization
        initialGripperJointValues = self.get_gripper_jpos()
        initialTime = self.mj_sim.data.time                  # starting time of gripper operation

        success = False

        # keep the history of finger joint position and velocity, this will be used to validate
        # whether finger is successfully closed
        # When finger is successfully closed and grasping is successful, jpos, jvel and touch sensor
        # curve should be flat and touch sensor reading should be close to finger close force
        time_log = []
        jpos_log = []
        jvel_log = []
        touch_log = [[], []]

        # get joint and actuator id for control
        g_qpos_id = self.mj_sim_param.gripper_qpos_id
        f_qpos_id = self.mj_sim_param.finger_qpos_id
        g_pos_ctrl_id = self.mj_sim_param.gripper_pos_ctrl_id
        g_vel_ctrl_id = self.mj_sim_param.gripper_vel_ctrl_id
        main_touch_id = self.mj_sim_param.finger_main_touch_dataid

        while True:
            # gravity compensation for gripper joints
            self.mj_sim.data.qfrc_applied[g_qpos_id] = self.mj_sim.data.qfrc_bias[g_qpos_id]
            # assign velocity and position actuator commands
            self.mj_sim.data.ctrl[g_pos_ctrl_id] = initialGripperJointValues
            self.mj_sim.data.ctrl[g_vel_ctrl_id] = np.zeros(initialGripperJointValues.shape)
            # assign gripper velocity actuator command, assign gripper grasping force
            self.mj_sim.data.qfrc_applied[f_qpos_id] = self.mj_sim.data.qfrc_bias[f_qpos_id] + force
            self.mj_simulation_step()
            # keep track of duration of this operation
            time_duration = self.mj_sim.data.time - initialTime
            # keep track of gripper joint value, velocity and main touch sensor reading
            time_log.append(time_duration)
            jpos_log.append(self.mj_sim.data.qpos[f_qpos_id][0])
            jvel_log.append(self.mj_sim.data.qvel[f_qpos_id][0])
            touch_log[0].append(self.mj_sim.data.sensordata[main_touch_id][2])
            touch_log[1].append(self.mj_sim.data.sensordata[main_touch_id][5])
            if len(jpos_log) > 10:  # only keep values of the past 20 steps
                jpos_log.pop(0)
                jvel_log.pop(0)
                touch_log[0].pop(0)
                touch_log[1].pop(0)
                time_log.pop(0)
                Y = np.array([jpos_log, jvel_log, touch_log[0]])
                X = np.array(time_log)
                slope = gripper_util.fit_slope(Y.transpose(), X)

                # check if gripper is fully closed
                # if gripper joint value, velocity and main touch sensor reading has stablized, that means gripper successfully closed
                if np.abs(slope)[0] < 1e-1 and np.abs(slope)[1] < 1e-1 and np.abs(slope)[2] < 1e-1 \
                        and touch_log[0][-1] + touch_log[1][-1] > force:
                    # print('Gripper successfully closed.\n Time duration for closing gripper is: {}\n'.format(time_duration))
                    success = True
                if time_duration > maxTime:
                    break
                    # print('[WARNING] Maximum time {}seconds reached while closing gripper.\n'.format(maxTime))
        # return success
        return True # temperal fix for training

    def open_gripper(self, force=1, maxTime=.4):
        # print('===========================================================================')
        # initialization
        initialGripperJointValues = self.get_gripper_jpos()
        initialTime = self.mj_sim.data.time                  # starting time of gripper operation
        time_log = []
        jpos_log = []
        jvel_log = []
        touch_log = [[], []]
        # get joint and actuator id for control
        g_qpos_id = self.mj_sim_param.gripper_qpos_id
        f_qpos_id = self.mj_sim_param.finger_qpos_id
        g_pos_ctrl_id = self.mj_sim_param.gripper_pos_ctrl_id
        g_vel_ctrl_id = self.mj_sim_param.gripper_vel_ctrl_id
        main_touch_id = self.mj_sim_param.finger_main_touch_dataid

        success = False

        while True:
            # gravity compensation
            self.mj_sim.data.qfrc_applied[g_qpos_id] = self.mj_sim.data.qfrc_bias[g_qpos_id]
            # assign velocity and position actuator commands
            self.mj_sim.data.ctrl[g_pos_ctrl_id] = initialGripperJointValues
            self.mj_sim.data.ctrl[g_vel_ctrl_id] = np.zeros(initialGripperJointValues.shape)
            # assign gripper grasping force
            self.mj_sim.data.qfrc_applied[f_qpos_id] = self.mj_sim.data.qfrc_bias[f_qpos_id] - force
            self.mj_simulation_step()
            # keep track of duration of this operation
            time_duration = self.mj_sim.data.time - initialTime
            # keep track of gripper joint value, velocity and main touch sensor reading
            time_log.append(time_duration)
            jpos_log.append(self.mj_sim.data.qpos[f_qpos_id][0])
            jvel_log.append(self.mj_sim.data.qvel[f_qpos_id][0])
            touch_log[0].append(self.mj_sim.data.sensordata[main_touch_id][2])
            touch_log[1].append(self.mj_sim.data.sensordata[main_touch_id][5])
            if len(jpos_log) > 10:  # only keep values of the past 20 steps
                jpos_log.pop(0)
                jvel_log.pop(0)
                touch_log[0].pop(0)
                touch_log[1].pop(0)
                time_log.pop(0)
                Y = np.array([jpos_log, jvel_log, touch_log[0]])
                X = np.array(time_log)
                slope = gripper_util.fit_slope(Y.transpose(), X)

                # check if gripper is fully closed
                # if gripper joint value, velocity and main touch sensor reading has stablized, that means gripper successfully opened
                if np.abs(slope)[0] < 1e-1 and np.abs(slope)[1] < 1e-1 and np.abs(slope)[2] < 1e-1 \
                        and touch_log[0][-1] + touch_log[1][-1] < force:
                    # print('Gripper successfully Opened.\n Time duration for opening gripper is: {}\n'.format(time_duration))
                    success = True
                if time_duration > maxTime:
                    break
                    # print('[WARNING] Maximum time {}seconds reached while opening gripper.\n'.format(maxTime))
        # return success
        return True # temperal fix for training

    """ Helper functions """
    def get_node_xpos(self):
        idx = self.mj_sim_param.velcro_node_bodyid
        return self.mj_sim.data.body_xpos[idx]

    def get_table_node_xpos(self):
    	idx = self.mj_sim_param.table_node_siteid
    	return self.mj_sim.data.site_xpos[idx]

    def get_tendon_lenngth(self):
        idx = self.mj_sim_param.velcro_tendon_id
        return self.mj_sim.data.ten_length[idx]

    def get_tendon_binary(self):
        idx = self.mj_sim_param.velcro_tendon_id
        stiffness = self.mj_sim.model.tendon_stiffness[idx]
        ret = stiffness > 0
        return ret.astype(int)

    def get_gripper_jpos(self):
        # joint values are in the order of: t_x, t_y, t_z, r_z, r_y, r_x
        idx = self.mj_sim_param.gripper_qpos_id
        return self.mj_sim.data.qpos[idx]

    def get_gripper_xpos(self):
        idx = self.mj_sim_param.gripper_qpos_id
        qpos = self.mj_sim.data.qpos[idx]
        lxmat = self.mj_sim.data.body_xmat[self.mj_sim.model._body_name2id['left_finger_skin']]
        rxmat = self.mj_sim.data.body_xmat[self.mj_sim.model._body_name2id['right_finger_skin']]
        return np.concatenate([qpos, lxmat, rxmat])   # shape 24

    def get_gripper_jvel(self):
        idx = self.mj_sim_param.gripper_qpos_id
        return self.mj_sim.data.qvel[idx]

    def get_finger_jpos(self):
        idx = self.mj_sim_param.finger_qpos_id
        return self.mj_sim.data.qpos[idx]

    def get_finger_jvel(self):
        idx = self.mj_sim_param.finger_qpos_id
        return self.mj_sim.data.qvel[idx]

    def get_ee_pose(self):
        gripperJointValues = self.get_gripper_jpos()
        theta = gripperJointValues[3:]   # values of 3 rotational joints r_x r_y r_z
        R = gripper_util.euler2rot(theta)
        P = gripperJointValues[0:3]  # vlaues of 3 sliding joints
        return gripper_util.pr2ee_pose(P, R)


    def read_main_touch(self):
        idx = self.mj_sim_param.finger_main_touch_dataid
        return self.mj_sim.data.sensordata[idx]

    def store_fbbuffer(self, main_touch, fpos, num_broken, tool_pose):
        self.feedback_buffer["main_touch"] = main_touch
        self.feedback_buffer["fpos"] = fpos
        self.feedback_buffer["num_broken"] = num_broken
        self.feedback_buffer["tool_pose"] = tool_pose

    def get_num_broken_buffer(self, hap_sample):
        if self.feedback_buffer["num_broken"] is None:
            self.feedback_buffer["num_broken"] = np.zeros((hap_sample, 1))
        return self.feedback_buffer["num_broken"].flatten()


    def get_main_touch_buffer(self, hap_sample):
        if self.feedback_buffer["main_touch"] is None:
            self.feedback_buffer["main_touch"] = np.zeros((hap_sample, 6))
        return self.feedback_buffer["main_touch"].flatten()

    def get_fpos_buffer(self, hap_sample):
        if self.feedback_buffer["fpos"] is None:
            self.feedback_buffer["fpos"] = np.zeros((hap_sample, 2))
        return self.feedback_buffer["fpos"].flatten()

    def get_touch_fpos_buffer(self, hap_sample):
        if self.feedback_buffer["main_touch"] is None:
            self.feedback_buffer["main_touch"] = np.zeros((hap_sample, 6))
        if self.feedback_buffer["fpos"] is None:
            self.feedback_buffer["fpos"] = np.zeros((hap_sample, 2))
        main_touch_flatten = self.feedback_buffer["main_touch"].flatten()
        fpos_flatten = self.feedback_buffer["fpos"].flatten()
        return np.hstack((main_touch_flatten, fpos_flatten))

    def get_all_touch_buffer(self, hap_sample):
        if self.feedback_buffer["main_touch"] is None:
            self.feedback_buffer["main_touch"] = np.zeros((hap_sample, 6))
        if self.feedback_buffer["num_broken"] is None:
            self.feedback_buffer["num_broken"] = np.zeros((hap_sample, 1))

        main_touch_flatten = self.feedback_buffer["main_touch"].flatten()
        num_broken_flatten = self.feedback_buffer["num_broken"].flatten()
        return np.hstack((main_touch_flatten, num_broken_flatten))