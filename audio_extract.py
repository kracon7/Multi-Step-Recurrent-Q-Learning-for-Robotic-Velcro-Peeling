import os
import sys
import argparse
import pyaudio
import wave
import numpy as np
import librosa
import matplotlib.pyplot as plt
import librosa.display
import datetime as dt
from itertools import compress
from utils.memory import ReplayMemory, Transition

def load_txt(path):
    file = open(path, 'r')
    result = []
    for line in file:
        result.append(line.rstrip())
    return result

def pose_str2list(str):
    pose = str.split(',')
    pose = [float(item) for item in pose]
    return pose

def sort_traj(traj_txt):
    '''
    This function convert the raw text log into a list object of each trajectory step
    Input: a list of rstripped text of trajectory text file
    Output: a list object 'traj' for each step, each step is a dict object, 
            which consists of start and end timestamp, EE pose; terminal/slip
    '''
    traj = []
    for i, line in enumerate(traj_txt):
        if 'action' in line:
            # extract time text and then convert to float timestamp
            ts_txt = line.split()[1]
            ts = timestamp(ts_txt)

            action = line.split()[3]
            if 'finished' in traj_txt[i+2]:
                # !!!!!!!!!!!!!!!!!!!!!!!!! start pose is now string, write a str to list function to 
                #  convert it to list object
                start_pose = traj_txt[i+1].split('[')[-1].split(']')[0]
                end_pose   = traj_txt[i+3].split('[')[-1].split(']')[0]

                start_pose = pose_str2list(start_pose)
                end_pose = pose_str2list(end_pose)
                # extract time text and then convert to float timestamp
                te_txt = traj_txt[i+2].split()[1]
                te = timestamp(te_txt)

                if 'Terminated!' in traj_txt[i+4]:
                    info = 't'
                elif 'Slipped!' in traj_txt[i+4]:
                    info = 's'
                else:
                    info = None
                step = {'ts': ts, 'te': te, 'p0': start_pose, 'p1': end_pose, 'info': info}
                traj.append(step)
            else:
                print('something is wrong, line {} action did not finish properly'. format(i))
    return traj

def sort_force(force_txt, traj):
    '''
    force_txt in format '19 13:17:11,36880   fx:  -0.240, fy:  -1.779, fz:  -0.592'
    return a list object 'force' with the same length as 'traj', each element of 'force' 
    is [t, [fx, fy, fz]]
    '''
    N = len(force_txt)
    len_traj = len(traj)
    k = 0
    untrimmed = []      # un-trimmed force log
    force = [[] for i in range(len_traj)]

    for i, line in enumerate(force_txt):
        t = timestamp(line.split()[1])
        fx = float(line.split()[3].split(',')[0])
        fy = float(line.split()[5].split(',')[0])
        fz = float(line.split()[7])
        f = [fx, fy, fz]
        untrimmed.append([t, f])

    for k in range(len_traj):
        # get start and end timestamp for the trajectory step
        step = traj[k]
        ts = step['ts']
        te = step['te']
        for i in range(N):
            t = untrimmed[i][0]
            if t > ts and t < te:
                # this line is within the traj step
                force[k].append(untrimmed[i])

    # get the num of force measurement for every traj step
    step_force_len = np.array([len(force[i]) for i in range(len_traj)])
    force = list(compress(force, step_force_len > 5))
    traj = list(compress(traj, step_force_len > 5))
    
    step_force_len = np.array([len(force[i]) for i, _ in enumerate(force)])
    std_len = np.amin(step_force_len)
    force = [force[i][:std_len] for i, _ in enumerate(force)]
    # force.pop(np.array(step_force_len).dtype() < 5)
    # traj.pop(np.array(step_force_len) < 5)
    
    return force


def txt2force(txt):
    '''
    force text in format 'fx:  -0.240, fy:  -1.779, fz:  -0.592'
    '''
    [fx, fy, fz] = txt.split(',')
    return [float(fx.split(':')[1]), float(fy.split(':')[1]),float(fz.split(':')[1])]


def timestamp(time_txt):
    '''
    time text is in format hour:min:sec,millisec

    '''
    millisec = time_txt.split(',')[1]
    hour, minute, second = time_txt.split(',')[0].split(':')
    return float(hour) * 3600 + float(minute) * 60 + float(second) + float(millisec) * 1e-7

def main(args):

    traj_txt = load_txt(args.traj_path)
    force_txt = load_txt(args.force_path)

    memory = ReplayMemory(500000)

    traj = sort_traj(traj_txt)
    force = sort_force(force_txt, traj)

    # x_1 , sr_1 = librosa.load(args.audio_path)

        # memory.push(state, action, next_state, reward)

    a = 1


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Record audio signal')
    parser.add_argument('--audio_path', default='/home/jc/logs/realrobot/batch_1/audio_1.wav', help='output file path')
    parser.add_argument('--traj_path', default='/home/jc/logs/realrobot/traj_log_1.txt', help='output file path')
    parser.add_argument('--force_path', default='/home/jc/logs/realrobot/force_1.txt', help='output file path')
    parser.add_argument('--n_channel', default=1, type=int, help='number of channels')
    parser.add_argument('--force_freq', default=25, type=int, help='number frequency of force measurement every traj step')
     

    args = parser.parse_args()
    
    main(args)
