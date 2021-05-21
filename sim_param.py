import numpy as np

class SimParameter():
    def __init__(self, mj_sim):
        # When initializing, extract qpos id and ctrl id
        # qpos id for check joint values and velocities
        # ctrl id for assigning commands to position/velocity actuators

        gripper_pos_actuator_names = ['trans_x_p', 'trans_y_p', 'trans_z_p', 'rot_x_p', 'rot_y_p', 'rot_z_p']  # gripper position actuators
        gripper_vel_actuator_names = ['trans_x_v', 'trans_y_v', 'trans_z_v', 'rot_x_v', 'rot_y_v', 'rot_z_v']  # gripper velocity actuators
        gripper_joint_names = ['trans_x', 'trans_y', 'trans_z', 'rot_x', 'rot_y', 'rot_z']                     # gripper joint names
        finger_pos_actuator_names = ['finger_p']             # finger position actuators
        finger_vel_actuator_names = ['finger_v']             # finger velocity actuators
        finger_joint_names = ['l_f', 'r_f']                                                                                    # finger joint names
        finger_main_touch_names = ['lf_main', 'rf_main']          # main touch sensors are used to check whether or not gripper is fully closed

        self.nJoints = len(gripper_joint_names)

        # extracting qpos id for joints
        self.gripper_qpos_id = []
        for name in gripper_joint_names:
            self.gripper_qpos_id.append(mj_sim.model._joint_name2id[name])                                    # in radians

        # extracting qpos id for gripper
        self.finger_qpos_id = []
        for name in finger_joint_names:
            self.finger_qpos_id.append(mj_sim.model._joint_name2id[name])

        # extracting ctrl id for arm position actuators
        self.gripper_pos_ctrl_id = []
        for name in gripper_pos_actuator_names:
            self.gripper_pos_ctrl_id.append(mj_sim.model._actuator_name2id[name])

        # extracting ctrl id for arm velocity actuators
        self.gripper_vel_ctrl_id = []
        for name in gripper_vel_actuator_names:
            self.gripper_vel_ctrl_id.append(mj_sim.model._actuator_name2id[name])

        # extracting ctrl id for gripper actuator
        self.finger_pos_ctrl_id = mj_sim.model._actuator_name2id[finger_pos_actuator_names[0]]
        self.finger_vel_ctrl_id = mj_sim.model._actuator_name2id[finger_vel_actuator_names[0]]

        # extracting gripper main touch sensor id
        self.finger_main_touch_dataid = []
        for name in finger_main_touch_names:
            sensorid = mj_sim.model._sensor_name2id[name]
            sensor_adr = mj_sim.model.sensor_adr[sensorid]
            sensor_dim = mj_sim.model.sensor_dim[sensorid]
            self.finger_main_touch_dataid += np.arange(sensor_adr, sensor_adr + sensor_dim).tolist()
        # self.finger_main_touch_dataid = [0,1,2,3,4,5]
        
        self.timeStep = mj_sim.model.opt.timestep   # simulation timestep

        # velcro tendon id and other properties, extrated initially and will be used later to
        # reset these tendons
        numX = 36
        numY = 6    # numY must be odd

        velcro_tendon_names = []
        for i in range(numX):
            for j in range(numY):
                tendonName = 'velcroTendon_{}_{}'.format(i, j)
                velcro_tendon_names.append(tendonName)

        handle_tendon_names = []
        for i in range(numY):
            tendonName = "handleTendon_{}".format(i)
            handle_tendon_names.append(tendonName)

        # extracting velcro tendon parameters
        self.velcro_tendon_names = velcro_tendon_names
        self.velcro_tendon_id = []
        self.velcro_tendon_stiff_init = []
        self.velcro_tendon_color_init = []
        if velcro_tendon_names is not None:
            for name in velcro_tendon_names:
                self.velcro_tendon_id.append(mj_sim.model._tendon_name2id[name])
        # extract default velcro tendon stiff and color, save to reset model later
        self.velcro_tendon_stiff_init = mj_sim.model.tendon_stiffness[self.velcro_tendon_id]
        self.velcro_tendon_color_init = mj_sim.model.tendon_rgba[self.velcro_tendon_id]

        self.handle_tendon_id = []
        if handle_tendon_names is not None:
            for name in handle_tendon_names:
                self.handle_tendon_id.append(mj_sim.model._tendon_name2id[name])

        # get velcro nodes name and id
        self.velcro_node_names=[]
        self.velcro_node_bodyid=[]
        self.velcro_node_jointid=[]
        for i in range(numX):
            for j in range(numY):
                node_name = 'comp_{}_{}'.format(i, j)
                idx = mj_sim.model._body_name2id[node_name]
                self.velcro_node_names.append(node_name)
                self.velcro_node_bodyid.append(idx)
                self.velcro_node_jointid.append(mj_sim.model._joint_name2id[node_name])



        # get table nodes name and id
        self.table_node_names=[]
        self.table_node_siteid=[]
        for i in range(numX):
            for j in range(numY):
                node_name = 'table_{}_{}'.format(i, j)
                idx = mj_sim.model._site_name2id[node_name]
                self.table_node_names.append(node_name)
                self.table_node_siteid.append(idx)