from utils.gripper_util import parameterize_xml
import numpy as np
import os
import pickle

cwd = os.getcwd()

output_dir = os.path.join(cwd, 'tests/test_xmls/')

num_case_1 = 200            # number of files generated in total
num_case_2 = 200           # number of files generated in total
num_case_3 = 200           # number of files generated in total

# case 1, translation and rotation wrt z-axis
case_1 = []
for i in range(num_case_1):
    geom = 'flat'
    origin_offset = 0.4*(np.array([np.random.rand(), np.random.rand(), 0.5]) -0.5)
    rot_mag = np.array([0, 0,  np.pi])
    euler = np.multiply(rot_mag, 2 * np.random.rand(3) -1)
    radius = 10

    case_1.append([geom, origin_offset, euler, radius])

with open(output_dir+'temp_1_1.pickle', 'wb') as f:
    pickle.dump(case_1, f)
    f.close()


# case 2, translation and rotation wrt x,y,z-axis
case_2 = []
for i in range(num_case_2):
    geom = 'flat'
    origin_offset = 0.4*(np.array([np.random.rand(), np.random.rand(), 0.5]) -0.5)
    rot_mag = np.array([np.pi / 4, np.pi / 4,  np.pi])
    euler = np.multiply(rot_mag, 2 * np.random.rand(3) -1)
    radius = 10

    case_2.append([geom, origin_offset, euler, radius])

with open(output_dir+'temp_1_2.pickle', 'wb') as f:
    pickle.dump(case_2, f)
    f.close()


# case 3, translation and rotation wrt x,y,z-axis
case_3 = []
for i in range(num_case_3):
    # select geom type
    velcro_geom = ['flat', 'cylinder']
    geom_idx = np.random.randint(0, len(velcro_geom))
    geom = velcro_geom[geom_idx]

    # translate origin
    origin_offset = 0.4*(np.array([np.random.rand(), np.random.rand(), 0.5]) -0.5)
    
    # rotation wrt x,y,z-axis
    rot_mag = np.array([np.pi / 4, np.pi / 4,  np.pi])
    euler = np.multiply(rot_mag, 2 * np.random.rand(3) -1)

    # radius of cylinder if necessary
    radius = 0.6 + 0.6 * np.random.rand()

    case_3.append([geom, origin_offset, euler, radius])

with open(output_dir+'temp_1_3.pickle', 'wb') as f:
    pickle.dump(case_3, f)
    f.close()

