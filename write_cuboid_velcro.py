# insert velcro model into an existing mujoco xml file
# This script generates the XML file for simulating a planar cloth in XY plane\

# The cloth is consist of spheres and tendons forming a streching grid
# Adjust numX, numY, xStep, yStep to change the size and resolution of the grid
# Adjust planeTendonStrength to change the planar tendon stiffness  
import os
import argparse
import numpy as np
import math
from utils import gripper_util


np.set_printoptions(suppress=True)


class PlaneVelcroWriter(object):
    def __init__(self, numX = 16, numY = 5, planeTendonStrength = 5000, velcroTendonStrength = 400, 
                    xOrigin = 0., yOrigin = 0., xStep = 0.04, yStep = 0.04):
        # numY must be odd
        self.numX = numX
        self.numY = numY  # numY must be odd
        self.planeTendonStrength = planeTendonStrength
        self.velcroTendonStrength = velcroTendonStrength
        self.xOrigin = xOrigin
        self.yOrigin = yOrigin
        self.xStep = xStep
        self.yStep = yStep

    def write_simple_tendon(self, siteName_1, siteName_2, tendonName, range = [0,1], stiffness = 1, damping=5, rgba = [0.5, 0.5, 0.5, 0]):
        ret = []
        ret.append('        <spatial name="{}" limited="true" range="{} {}" stiffness="{}" width="0.003" damping="{}" \
            rgba="{} {} {} {}" >\n'.format(tendonName, range[0], range[1], stiffness, damping, rgba[0], rgba[1], rgba[2], rgba[3]))
        ret.append('            <site site="{}" />\n'.format(siteName_1))
        ret.append('            <site site="{}" />\n'.format(siteName_2))
        ret.append('        </spatial>\n')
        return ret

    def write_plane_table(self):
        ret = []
        # table body and table sites
        ret.append('    <body name="table" pos="0 0 0" >\n')

        # handle and sites of the handle
        handle_pos = np.array([-0.06 + self.xOrigin, (self.numY-1)/2 * self.yStep + self.yOrigin, 0.02])
        rotated_handle_pos = handle_pos
        ret.append('        <body name="handle" pos="{} {} {}" >\n'.format(rotated_handle_pos[0], rotated_handle_pos[1], rotated_handle_pos[2]))
        ret.append('            <joint name="handle_x" type="slide" pos="0 0 0" axis="1 0 0" damping="10" />\n')
        ret.append('            <joint name="handle_y" type="slide" pos="0 0 0" axis="0 1 0" damping="10" />\n')
        ret.append('            <joint name="handle_z" type="slide" pos="0 0 0" axis="0 0 1" damping="10" />\n')
        ret.append('            <joint name="handle_rot_x" type="hinge" pos="0 0 0" axis="1 0 0" damping="10" />\n')
        ret.append('            <joint name="handle_rot_y" type="hinge" pos="0 0 0" axis="0 1 0" damping="10" />\n')
        ret.append('            <joint name="handle_rot_z" type="hinge" pos="0 0 0" axis="0 0 1" damping="10" />\n')
        handleLength = (self.numY+1) / 2 * self.yStep
        ret.append('            <geom name="handle" material="bench_mat" type="box" size="0.04 {} 0.01" rgba="1 0 0 1" density="50" />\n'.format(handleLength))
        # sites of handle
        for i in range(self.numY):
            handleSiteName = 'handle_{}'.format(i)
            yPos = (i-(self.numY-1)/2) * self.yStep
            ret.append('            <site name=\'{}\' type=\'sphere\' size=\'0.01\' pos=\'0 {} 0\' rgba=\'0 1 0 1\' />\n'.format(handleSiteName, yPos))

        ret.append('            <body name="handle2" pos="0 0 0.06" >\n')
        # ret.append('                <joint type="hinge" axis="0 0 1" damping="3" />\n')
        ret.append('                <geom name="handle2" material="bench_mat" pos="0 0 0.05" type="box" size="0.01 0.02 0.08" density="50" rgba="1 0 0 1"/>\n')
        ret.append('            </body>\n')              # end of handle2
        ret.append('        </body>\n')                  # end of handle

        table_pos = np.array([self.xOrigin, self.yOrigin, -0.49])
        rotated_table_pos = table_pos
        ret.append('        <body name="tabletop" pos="{} {} {}" >\n'.format(rotated_table_pos[0], rotated_table_pos[1], rotated_table_pos[2]))
        table_width = int(self.numX / 2) * self.xStep
        ret.append('            <geom name="tabletop" size="{} {} 0.49" type="box" rgba="0.4 0.4 0.4 1" />\n'.format(
                                table_width, table_width))
        ret.append('        </body>\n')
        for i in range(int(self.numX / 2)):
            for j in range(self.numY):
                tableSiteName = 'table_{}_{}'.format(i,j)  # example: table_1_1
                x = self.xOrigin + i * self.xStep
                y = self.yOrigin + j * self.yStep
                z = 0
                pos_init = np.array([x, y, z])
                pos_trans = pos_init
                ret.append('        <site name="{}" type="sphere" size="0.005" pos="{} {} {}" rgba="1 1 1 1" />\n'.format(
                                tableSiteName, pos_trans[0], pos_trans[1], pos_trans[2]))
        for i in range(int(self.numX / 2), self.numX):
            for j in range(self.numY):
                tableSiteName = 'table_{}_{}'.format(i,j)  # example: table_1_1
                x = table_width
                y = self.yOrigin + j * self.yStep
                z = 0 - (i - int(self.numX / 2)) * self.xStep
                pos_init = np.array([x, y, z])
                pos_trans = pos_init
                ret.append('        <site name="{}" type="sphere" size="0.005" pos="{} {} {}" rgba="1 1 1 1" />\n'.format(
                                tableSiteName, pos_trans[0], pos_trans[1], pos_trans[2]))
        ret.append('    </body>\n\n')
        return ret

    def write_plane_composite(self):
        ret = []
        # Composite body (spheres)
        for i in range(int(self.numX / 2)):
            for j in range(self.numY):
                sphereName = 'comp_{}_{}'.format(i, j)
                siteName = 'comp_{}_{}_0'.format(i, j)
                x = self.xOrigin + i * self.xStep
                y = self.yOrigin + j * self.yStep
                z = 0.015
                pos_init = np.array([x, y, z])
                pos_trans = pos_init
                ret.append('        <body name="{}" pos="{} {} {}" >\n'.format(sphereName, pos_trans[0], pos_trans[1], pos_trans[2]))
                ret.append('            <joint type="free" name="{}" />\n'.format(sphereName))
                ret.append('            <geom type="sphere" size="0.001" density="50" rgba="1 1 1 1" />\n')
                ret.append('            <site name="{}" type="sphere" size="0.005" />\n'.format(siteName))
                ret.append('        </body>\n') 

        table_width = int(self.numX / 2) * self.xStep
        for i in range(int(self.numX/2), self.numX):
            for j in range(self.numY):
                sphereName = 'comp_{}_{}'.format(i, j)
                siteName = 'comp_{}_{}_0'.format(i, j)
                x = table_width + 0.015
                y = self.yOrigin + j * self.yStep
                z = 0 - (i - int(self.numX / 2)) * self.xStep
                pos_init = np.array([x, y, z])
                pos_trans = pos_init
                ret.append('        <body name="{}" pos="{} {} {}" >\n'.format(sphereName, pos_trans[0], pos_trans[1], pos_trans[2]))
                ret.append('            <joint type="free" name="{}" />\n'.format(sphereName))
                ret.append('            <geom type="sphere" size="0.001" density="50" rgba="1 1 1 1" />\n')
                ret.append('            <site name="{}" type="sphere" size="0.005" />\n'.format(siteName))
                ret.append('        </body>\n')  
        return ret

    def write_tendons(self):
        ret = []
        # tendon part
        for i in range(self.numX):
            for j in range(self.numY):
                # tendons for interconnection of composite body
                if i < self.numX-1:
                    siteName_1 = 'comp_{}_{}_0'.format(i, j)
                    siteName_2 = 'comp_{}_{}_0'.format(i+1, j)
                    tendonName = 'planeTendon_{}_{}__{}_{}'.format(i,j,i+1,j)
                    ret[0:0] = self.write_simple_tendon(siteName_1, siteName_2, tendonName, range=[0, 0.045], stiffness = self.planeTendonStrength, damping=30)
                if j < self.numY-1:
                    siteName_1 = 'comp_{}_{}_0'.format(i, j)
                    siteName_2 = 'comp_{}_{}_0'.format(i, j+1)
                    tendonName = 'planeTendon_{}_{}__{}_{}'.format(i,j,i,j+1)
                    ret[0:0] = self.write_simple_tendon(siteName_1, siteName_2, tendonName, range=[0, 0.045], stiffness = self.planeTendonStrength, damping=30)

                # tendons for connection between table sites and composite body
                # no damping for these tendons
                siteName_1 = 'comp_{}_{}_0'.format(i, j)
                siteName_2 = 'table_{}_{}'.format(i, j)
                tendonName = 'velcroTendon_{}_{}'.format(i,j)
                ret[0:0] = self.write_simple_tendon(siteName_1, siteName_2, tendonName, range=[0, 20], stiffness = self.velcroTendonStrength, damping=.2, rgba=[0,1,0,1])
        return ret

    def write_handle_tendons(self):
        ret = []
        # tendons for connection between handle and composite body
        for i in range(self.numY):
            siteName_1 = 'handle_{}'.format(i)
            siteName_2 = 'comp_0_{}_0'.format(i)
            tendonName = 'handleTendon_{}'.format(i)
            ret[0:0] = self.write_simple_tendon(siteName_1, siteName_2, tendonName, range=[-0.5, 0.065], stiffness=5e4,  damping=30)
        return ret     


def main(args):

    # read out all lines in the target XML file and insert the velcro models
    capturedOut = []
    load_file = open(args.load_path, 'r')
    for line in load_file:
        capturedOut.append(line)
    load_file.close()

    # generate velcro model lines
    velcroWriter = PlaneVelcroWriter(numX = args.n_x, numY = args.n_y, planeTendonStrength = args.plane_tendon_strength, 
                velcroTendonStrength = args.velcro_tendon_strength, xOrigin = args.o_x, yOrigin = args.o_y, 
                xStep = args.d_x, yStep = args.d_y)
    
    table = velcroWriter.write_plane_table()
    velcro = velcroWriter.write_plane_composite()
    tendons = velcroWriter.write_tendons()
    handleTendons = velcroWriter.write_handle_tendons()

    # insert velcro model lines into capturedOut
    for i, line in enumerate(capturedOut):
        keyWord = '<!-- Insert table here! -->'
        if line.find(keyWord) >= 0:
            pos = i + 1 
            capturedOut[pos:pos] = table
    for i, line in enumerate(capturedOut):
        keyWord = '<!-- Insert velcro here! -->'
        if line.find(keyWord) >= 0:
            pos = i + 1 
            capturedOut[pos:pos] = velcro
    for i, line in enumerate(capturedOut):
        keyWord = '<!-- Insert tendons here! -->'
        if line.find(keyWord) >= 0:
            pos = i + 1 
            capturedOut[pos:pos] = handleTendons
            capturedOut[pos:pos] = tendons

    # write the updated lines to a new file
    with open(args.output_path, 'w') as f:
        for line in capturedOut:
            f.write(line)
    f.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Inset velcro part to paraGripper xml file')
    parser.add_argument('--load_path', default='./models/paraGripper.xml', help='XML model to load')
    parser.add_argument('--output_path', default='./models/cuboid_velcro.xml', help='path where to save')
    parser.add_argument('--n_x', default=36, type=int, help='length of velcro')
    parser.add_argument('--n_y', default=6, type=int, help='width of velcro')
    parser.add_argument('--d_x', default=0.02, type=float, help='step size of velcro in x')
    parser.add_argument('--d_y', default=0.02, type=float, help='step size of velcro in y')
    parser.add_argument('--o_x', default=0, type=float, help='origin of velcro in x')
    parser.add_argument('--o_y', default=0, type=float, help='origin of velcro in y')
    parser.add_argument('-p', '--plane_tendon_strength', default=120000, type=int)
    parser.add_argument('-v', '--velcro_tendon_strength', default=600, type=int)
    
    args = parser.parse_args()
    main(args)