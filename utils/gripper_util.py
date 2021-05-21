import numpy as np
import numpy.linalg as la
import math
from math import sin, cos
import os
from mujoco_py import MjViewer, load_model_from_path, MjSim

def load_model(model_path, render=True):
    model = load_model_from_path(model_path)
    sim = MjSim(model)
    if render:
        viewer = MjViewer(sim)
    else:
        viewer = None
    return model, sim, viewer

def init_model(sim, geom = True, trans = True, rot = True):
    velcro_geom = ['flat', 'cylinder']
    if geom:
        geom = velcro_geom[np.random.randint(0, len(velcro_geom))]
    else:
        geom = 'flat'
    
    if trans:
        origin_offset = 0.4*(np.array([np.random.rand(), np.random.rand(), 0.5]) -0.5)
    else:
        origin_offset = np.zeros(3)
    
    if rot:
        rot_mag = np.array([np.pi / 4, np.pi / 4,  np.pi])
        euler = np.multiply(rot_mag, 2 * np.random.rand(3) -1)
    else:
        euler = np.zeros(3)

    if geom is 'cylinder':
        radius = 0.6 + 0.6 * np.random.rand()
    else:
        radius = 10
    change_sim(sim, geom, origin_offset, euler, radius)
    return [geom, origin_offset, euler, radius]

def change_sim(sim, geom, origin_offset=[0.,0.,0.], euler=[0.,0.,0.], radius=0.5):
    tabletop_bodyid = sim.model._body_name2id['tabletop']
    tabletop_geomid = sim.model._geom_name2id['tabletop']
    if geom == 'flat':
        sim.model.body_pos[tabletop_bodyid][:] = np.array([0,0,-0.49])
        sim.model.body_quat[tabletop_bodyid][:] = np.array([1., 0., 0. , 0.])
        sim.model.geom_type[tabletop_geomid] = 6
        sim.model.geom_size[tabletop_geomid][:] = np.array([2, 2, 0.49])
        rot_flat_model(sim, origin_offset, euler)
    elif geom == 'cylinder':
        sim.model.body_pos[tabletop_bodyid][:] = np.array([0,0,-radius])
        sim.model.body_quat[tabletop_bodyid][:] = np.array([0.70710678118, 0.70710678118, 0. , 0.])
        sim.model.geom_type[tabletop_geomid] = 5
        sim.model.geom_size[tabletop_geomid][0] = radius 
        sim.model.geom_size[tabletop_geomid][1] = 1.5
        rot_cylnd_model(sim, origin_offset, euler, radius)
    else:
        print('Unrecognized geom type')

def rot_flat_model(sim, origin_offset=[0.,0.,0.], euler=[0.,0.,0.], numX=36, numY=6, xStep = 0.02, yStep = 0.02):
    # Translation for whole table object including table sites and tabletop geom
    # First translation and then rotation
    sim.model.body_pos[sim.model._body_name2id['table']] = origin_offset
    sim.model.body_quat[sim.model._body_name2id['table']] = rot_table(euler)
    sim.model.body_quat[sim.model._body_name2id['handle2']] = rot_handle2(euler)

    for i in range(numX):
        for j in range(numY):
            tableSiteName = 'table_{}_{}'.format(i,j)  # example: table_1_1
            compSiteName = 'comp_{}_{}'.format(i,j)  # example: comp_1_1
            x = i * xStep
            y = j * yStep
            z = 0
            
            pos = np.array([x, y, z])
            # reset table site pos in frame "tabletop", no need to rotate
            sim.model.site_pos[sim.model._site_name2id[tableSiteName]] = pos
            # reset comp body pos in frame "world", modify qpos
            jntid = sim.model._joint_name2id[compSiteName]
            qposadr = sim.model.jnt_qposadr[jntid]
            pos[2] += 0.015
            sim.data.qpos[qposadr: qposadr+3] = rot_comp(euler, pos)  + origin_offset
            sim.model.qpos0[qposadr: qposadr+3] = rot_comp(euler, pos)+ origin_offset
   
def rot_cylnd_model(sim, origin_offset=[0.,0.,0.], euler=[0.,0.,0.], radius=0.6, numX=36, numY=6, xStep = 0.02, yStep = 0.02):
    # Translation for whole table object including table sites and tabletop geom
    # First translation and then rotation
    sim.model.body_pos[sim.model._body_name2id['table']] = origin_offset
    # Rotate table and handle so that handle is point up
    sim.model.body_quat[sim.model._body_name2id['table']] = rot_table(euler)
    sim.model.body_quat[sim.model._body_name2id['handle2']] = rot_handle2(euler)
    
    # get table site pos in frame "tabletop", no need to rotate
    d_theta = xStep/radius
    for i in range(numX):
        for j in range(numY):
            tableSiteName = 'table_{}_{}'.format(i,j)  # example: table_1_1
            # anglar value on cylinder surface: self.d_theta  
            d_z = math.cos(i * d_theta) * radius
            d_x = math.sin(i * d_theta) * radius
            x = d_x 
            y = j * yStep
            z = 0 - radius + d_z
            sim.model.site_pos[sim.model._site_name2id[tableSiteName]] = np.array([x, y, z])
    
    # set comp body pos by modifying sim.data.qpos and sim.model.qpos0
    for i in range(numX):
        for j in range(numY):
            compSiteName = 'comp_{}_{}'.format(i,j)  # example: comp_1_1
            # anglar value on cylinder surface: self.d_theta  
            d_z = math.cos(i * d_theta) * (radius + 0.015)
            d_x = math.sin(i * d_theta) * (radius + 0.015)
            x = d_x 
            y = j * yStep
            z = 0 - radius + d_z
            
            pos = np.array([x, y, z])
            jntid = sim.model._joint_name2id[compSiteName]
            qposadr = sim.model.jnt_qposadr[jntid]
            sim.data.qpos[qposadr: qposadr+3] = rot_comp(euler, pos)+ origin_offset
            sim.model.qpos0[qposadr: qposadr+3] = rot_comp(euler, pos)+ origin_offset
 
def norm_img(img):
    '''
    Raw rgb image from mujoco has shape (w, h, 3)
    This function reshape it into (3, w, h) and normalize the value to [0,1]
    '''
    if img.shape[2] == 3:
        img = np.transpose(img, (2,0,1))
    img.astype('float')
    return img / 255

def norm_depth(depth):
    depth.astype('float')
    return depth / 10


def parameterize_xml(load_path, output_path, geom = 'flat', trans = np.zeros(2), rot = np.zeros(3), radius = 0.):
    # generate xml files from parameters
    cmd = 'python velcro_insert_oriented.py' + ' --load_path=' + load_path + ' --output_path=' + output_path
    cmd = cmd + ' --velcro_type=' + geom + ' --o_x={} --o_y={}'.format(trans[0], trans[1]) + \
            ' --theta_z={} --theta_y={} --theta_x={} --radius={}'.format(rot[2], rot[1], rot[0], radius)
    os.system(cmd)


def roll_pitch_yaw(theta, euler_seq='xyz'):
    gamma = theta[0]
    beta = theta[1]
    alpha = theta[2]
    cg = cos(gamma)
    sg = sin(gamma)
    cb = cos(beta)
    sb = sin(beta)
    ca = cos(alpha)
    sa = sin(alpha)
    r_x = np.array([[1, 0,  0  ],
                    [0, cg, -sg],
                    [0, sg, cg]])
    r_y = np.array([[cb, 0,  sb],
                    [0,  1, 0],
                    [-sb, 0, cb]])
    r_z = np.array([[ca, -sa,  0],
                    [sa, ca,   0],
                    [0,   0,   1]])
    if euler_seq == 'zyx':
        R = np.dot(r_z, np.dot(r_y, r_x))
    elif euler_seq == 'xyz':
        R = np.dot(r_x, np.dot(r_y, r_z))
    return R

def inverse_rpy(theta, euler_seq='xyz'):
    R = roll_pitch_yaw(theta, euler_seq = euler_seq)
    return rot2euler(R.T, euler_seq = euler_seq)

def pq2ee_pose(toolPosition, toolQuaternion):
    # convert tool position and quaternion to 3x4 ndarray ee_pose
    R = quaternion2Rotation(toolQuaternion)
    ee_pose = np.zeros((3,4))
    ee_pose[:,0:3] = R
    ee_pose[0,3] = toolPosition[0]
    ee_pose[1,3] = toolPosition[1]
    ee_pose[2,3] = toolPosition[2]
    return ee_pose

def pr2ee_pose(P, R):
    ee_pose = np.zeros((3,4))
    ee_pose[:,0:3] = R
    ee_pose[0,3] = P[0]
    ee_pose[1,3] = P[1]
    ee_pose[2,3] = P[2]
    return  ee_pose

def ee_pose2pq(ee_pose):
    # convert ee_pose into toolPosition and toolQuaternion
    R = ee_pose[:, 0:3]
    toolQuaternion = rotation2Quaternion(R)
    toolPosition = np.zeros(3)
    toolPosition[0] = ee_pose[0,3]
    toolPosition[1] = ee_pose[1,3]
    toolPosition[2] = ee_pose[2,3]
    return toolPosition, toolQuaternion

def ee_pose2pr(ee_pose):
    R = ee_pose[:, 0:3]
    P = np.zeros(3)
    P[0] = ee_pose[0,3]
    P[1] = ee_pose[1,3]
    P[2] = ee_pose[2,3]
    return P, R

def ee_pose2jpos(ee_pose):
    jointValues = np.zeros(6)
    P, R = ee_pose2pr(ee_pose)
    jointValues[0:3] = P
    theta = rot2euler(R)
    jointValues[3:] = theta
    return jointValues


# Checks if a matrix is a valid rotation matrix.
def isRotm(R):
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6

# Convert quaternion to rotation matrix
def quaternion2Rotation(q):
    qw = q[0]
    qx = q[1]
    qy = q[2]
    qz = q[3]
    R = np.array([[1 - 2 * qy ** 2 - 2 * qz ** 2, 2 * qx * qy - 2 * qz * qw, 2 * qx * qz + 2 * qy * qw],
                  [2 * qx * qy + 2 * qz * qw, 1 - 2 * qx ** 2 - 2 * qz ** 2, 2 * qy * qz - 2 * qx * qw],
                  [2 * qx * qz - 2 * qy * qw, 2 * qy * qz + 2 * qx * qw, 1 - 2 * qx ** 2 - 2 * qy ** 2]])
    if isRotm(R):
        return R
    else:
        raise Exception('R is not a rotation matrix, please check your quaternions')


# Convert rotation matrix to quaternion
def rotation2Quaternion(R):
    assert(isRotm(R))
    r11 = R[0][0]
    r12 = R[0][1]
    r13 = R[0][2]
    r21 = R[1][0]
    r22 = R[1][1]
    r23 = R[1][2]
    r31 = R[2][0]
    r32 = R[2][1]
    r33 = R[2][2]

    idx = np.array([r11+r22+r33, r11, r22, r33]).argmax()

    # computing four sets of solutions 
    if idx == 0:
        qw_1 = math.sqrt(1 + r11 + r22 + r33)
        q = 1/2 * np.array([qw_1,
                             (r32-r23)/qw_1,
                             (r13-r31)/qw_1,
                             (r21-r12)/qw_1
                             ])
    elif idx == 1:
        qx_2 = math.sqrt(1 + r11 - r22 - r33)
        q = 1/2 * np.array([(r32-r23)/qx_2,
                             qx_2,
                             (r12+r21)/qx_2,
                             (r31+r13)/qx_2
                             ])
    elif idx == 2:
        qy_3 = math.sqrt(1 - r11 + r22 - r33)
        q = 1/2 * np.array([(r13-r31)/qy_3,
                             (r12+r21)/qy_3,
                             qy_3,
                             (r23+r32)/qy_3
                             ])
    elif idx == 3:
        qz_4 = math.sqrt(1 - r11 - r22 + r33)
        q = 1/2* np.array([(r21-r12)/qz_4,
                            (r31+r13)/qz_4,
                            (r32+r23)/qz_4,
                            qz_4
                            ])

    if (la.norm(q) - 1) < 1e-3:
        return q
    else:
        raise Exception('Quaternion is not normalized, please check your rotation matrix')



def fit_slope(Y, X):
    # Y is a list of numbers, could be joint values or sensor readings
    # X is time
    z = np.polyfit(X, Y, 1)
    slope = z[0,:]
    return slope

def Normalize(V):
    # Normalizes a vector
    return V / np.linalg.norm(V)

def NearZero(z):
    # Determines whether a scalar is small enough to be treated as zero
    return abs(z) < 1e-6

def VecToso3(omg):
    """Converts a 3-vector to an so(3) representation
    :param omg: A 3-vector
    :return: The skew symmetric representation of omg
"""
    return np.array([[0,      -omg[2],  omg[1]],
                     [omg[2],       0, -omg[0]],
                     [-omg[1], omg[0],       0]])

def so3ToVec(so3mat):
    """Converts an so(3) representation to a 3-vector
    :param so3mat: A 3x3 skew-symmetric matrix
    :return: The 3-vector corresponding to so3mat
    """
    return np.array([so3mat[2][1], so3mat[0][2], so3mat[1][0]])

def AxisAng3(expc3):
    """Converts a 3-vector of exponential coordinates for rotation into
    axis-angle form

    :param expc3: A 3-vector of exponential coordinates for rotation
    :return omghat: A unit rotation axis
    :return theta: The corresponding rotation angle
    """
    return (Normalize(expc3), np.linalg.norm(expc3))

def MatrixExp3(so3mat):
    """Computes the matrix exponential of a matrix in so(3)

    :param so3mat: A 3x3 skew-symmetric matrix
    :return: The matrix exponential of so3mat
    """
    omgtheta = so3ToVec(so3mat)
    if NearZero(np.linalg.norm(omgtheta)):
        return np.eye(3)
    else:
        theta = AxisAng3(omgtheta)[1]
        omgmat = so3mat / theta
        return np.eye(3) + np.sin(theta) * omgmat \
               + (1 - np.cos(theta)) * np.dot(omgmat, omgmat)

def MatrixLog3(R):
    """Computes the matrix logarithm of a rotation matrix

    :param R: A 3x3 rotation matrix
    :return: The matrix logarithm of R
    """
    acosinput = (np.trace(R) - 1) / 2.0
    if acosinput >= 1:
        return np.zeros((3, 3))
    elif acosinput <= -1:
        if not NearZero(1 + R[2][2]):
            omg = (1.0 / np.sqrt(2 * (1 + R[2][2]))) \
                  * np.array([R[0][2], R[1][2], 1 + R[2][2]])
        elif not NearZero(1 + R[1][1]):
            omg = (1.0 / np.sqrt(2 * (1 + R[1][1]))) \
                  * np.array([R[0][1], 1 + R[1][1], R[2][1]])
        else:
            omg = (1.0 / np.sqrt(2 * (1 + R[0][0]))) \
                  * np.array([1 + R[0][0], R[1][0], R[2][0]])
        return VecToso3(np.pi * omg)
    else:
        theta = np.arccos(acosinput)
        return theta / 2.0 / np.sin(theta) * (R - np.array(R).T)

def sampleTrajectoryCartesian(P, R, P_target, R_target, resolution = 5e-3):
    dist = np.abs(np.subtract(P, P_target))
    nStep = int(np.amax(dist) / resolution)
    if nStep > 5:
        scaledTime = []
        cartesianTrajectory_P = []
        cartesianTrajectory_R = []
        for i in range(nStep + 1) :
            ST = QuinticTimeScaling(nStep, i)
            P_new = P + (P_target - P) * ST
            R_new = np.dot(R, MatrixExp3(MatrixLog3(np.dot(R.transpose(), R_target)) * ST))
            cartesianTrajectory_P.append(P_new)
            cartesianTrajectory_R.append(R_new)
        return True, cartesianTrajectory_P, cartesianTrajectory_R
    else:
        return False, []

def QuinticTimeScaling(Tf, t):
    # t in [0,1], return ratio of scaled time in [0,1]
    return 10 * (1.0 * t/Tf) ** 3 - 15 * (1.0 * t/Tf) ** 4 \
        + 6 * (1.0 * t/Tf) ** 5

def p2p_trajectory(startPoint, endPoint, resolution=1e-3):
    # start and end points are points in joint space. np array of shape (nv, ) (nv is num of joints)
    # resolution is used to calculate num of steps for trajectory
    #
    # return bool ret and list traj
    # if max distance in joint space coordinate is smaller than 5*resolution, ret = False and no trajectory
    # will be returned. Otherwise return ret = True and smapled trajectory
    # traj length nStep+1, each element is ndarray of shape (nv, )
    #
    nv = startPoint.shape[0]
    dist = np.abs(np.subtract(startPoint, endPoint, dtype = np.float64))
    nStep = int(np.amax(dist) / resolution)
    if nStep > 5:
        scaledTime = []
        traj = [startPoint]
        for i in range(nStep+1):
            ST = QuinticTimeScaling(nStep, i)
            scaledTime.append(ST)
            traj.append(startPoint + ST * (endPoint - startPoint))
        return True, traj
    else:
        return False, []

def get_vel_command(traj, timeStep):
    # given traj of length nStep+1, calculate velocity for each time step
    nStep = len(traj) -1
    velocityCommand = []
    for i in range(nStep):
        velocityCommand.append((traj[i+1] - traj[i]) / timeStep)
    return velocityCommand

def euler2rot(theta):
    # euler angles x, y, z
    R_x = np.array([[1,         0,                  0                   ],
                    [0,         cos(theta[0]), -sin(theta[0]) ],
                    [0,         sin(theta[0]), cos(theta[0])  ]])
    R_y = np.array([[cos(theta[1]),    0,      sin(theta[1])  ],
                    [0,                     1,      0                   ],
                    [-sin(theta[1]),   0,      cos(theta[1])  ]])
    R_z = np.array([[cos(theta[2]),    -sin(theta[2]),    0],
                    [sin(theta[2]),    cos(theta[2]),     0],
                    [0,                     0,                      1]])
    R = np.dot(np.dot(R_x, R_y), R_z)
    return R

def rot2euler(R, euler_seq = 'xyz'):
    # R = r_x * r_y * r_z
    assert(isRotm(R))
    if euler_seq == 'xyz':
        cy = math.sqrt(R[0,0] * R[0,0] +  R[0,1] * R[0,1])
        singular = cy < 1e-6

        if  not singular :
            x = math.atan2(-R[1,2] , R[2,2])
            y = math.atan2(R[0,2], cy)
            z = math.atan2(-R[0,1], R[0,0])
        else :
            x = math.atan2(R[2,1], R[1,1])
            y = math.atan2(R[0, 2], cy)
            z = 0

    elif euler_seq == 'zyx':
        sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
        singular = sy < 1e-6

        if  not singular :
            x = math.atan2(R[2,1] , R[2,2])
            y = math.atan2(-R[2,0], sy)
            z = math.atan2(R[1,0], R[0,0])
        else :
            x = math.atan2(-R[1,2], R[1,1])
            y = math.atan2(-R[2,0], sy)
            z = 0

    return np.array([x, y, z])


def rot_table(euler):
    x,y,z = euler
    R_x = np.array([[ 1,       0,       0      ],
                    [ 0,       cos(x), -sin(x) ],
                    [ 0,       sin(x),  cos(x) ]])
    R_y = np.array([[ cos(y),  0,       sin(y) ],
                    [ 0,       1,       0      ],
                    [-sin(y),  0,       cos(y) ]])
    R_z = np.array([[ cos(z), -sin(z),  0],
                    [ sin(z),  cos(z),  0],
                    [ 0,       0,       1]])
    rot = R_y @ R_x @ R_z
    return rotation2Quaternion(rot)

def rot_handle2(euler):
    x,y,z = euler
    R_x = np.array([[ 1,       0,       0      ],
                    [ 0,       cos(x), -sin(x) ],
                    [ 0,       sin(x),  cos(x) ]])
    R_y = np.array([[ cos(y),  0,       sin(y) ],
                    [ 0,       1,       0      ],
                    [-sin(y),  0,       cos(y) ]])
    R_z = np.array([[ cos(z), -sin(z),  0],
                    [ sin(z),  cos(z),  0],
                    [ 0,       0,       1]])
    rot = R_z.T @ R_x.T @ R_y.T  
    return rotation2Quaternion(rot)

def rot_comp(euler, pos):
    '''
    euler: euler angle theta_x, theta_y, theta_z
    pos: translation in world frame
    '''
    x,y,z = euler
    R_x = np.array([[ 1,       0,       0      ],
                    [ 0,       cos(x), -sin(x) ],
                    [ 0,       sin(x),  cos(x) ]])
    R_y = np.array([[ cos(y),  0,       sin(y) ],
                    [ 0,       1,       0      ],
                    [-sin(y),  0,       cos(y) ]])
    R_z = np.array([[ cos(z), -sin(z),  0],
                    [ sin(z),  cos(z),  0],
                    [ 0,       0,       1]])
    rot = R_y @ R_x @ R_z
    pos_trans = rot @ pos 
    return pos_trans

def init_for_test(sim, geom = True, trans = True, rot_z = True, rot_xy = True):
    velcro_geom = ['flat', 'cylinder']
    if geom:
        geom = velcro_geom[np.random.randint(0, len(velcro_geom))]
    else:
        geom = 'flat'
    
    if trans:
        origin_offset = 0.4*(np.array([np.random.rand(), np.random.rand(), 0.5]) -0.5)
    else:
        origin_offset = np.zeros(3)

    if rot_z:
        rot_mag = np.array([0, 0, np.pi])    
        if rot_xy:
            rot_mag = np.array([np.pi / 4, np.pi / 4,  np.pi])
        euler = np.multiply(rot_mag, 2 * np.random.rand(3) -1)
    else:
        euler = np.zeros(3)

    if geom is 'cylinder':
        radius = 0.6 + 0.6 * np.random.rand()
    else:
        radius = 10
    change_sim(sim, geom, origin_offset, euler, radius)
    return [geom, origin_offset, euler, radius]