import numpy as np
def load_txt(path):
    file = open(path, 'r')
    result = []
    for line in file:
        result.append(line.rstrip())
    return result

def timestamp(time_txt):
    '''
    time text is in format hour:min:sec,millisec

    '''
    millisec = time_txt.split(',')[1]
    hour, minute, second = time_txt.split(',')[0].split(':')
    return float(hour) * 3600 + float(minute) * 60 + float(second) + float(millisec) * 1e-7

def add_timestamp(t0, dt):
    '''
    t0 in timestamp format
    dt is str with format: 'minute:second:microsec'
    return timestamp
    '''
    minute, second, microsec = dt.split(':')
    return t0 + float(minute)*60 + float(second) + float(microsec)*1e-7

def pose_str2list(str):
    pose = str.split(',')
    pose = [float(item) for item in pose]
    return pose

def extrat_force(txt):
    result = []
    for i, line in enumerate(txt):
        try:
            items = line.split()
            time = timestamp(items[2])
            fx = float(items[4])
            fy = float(items[5])
            fz = float(items[6])
            result.append([time, fx, fy, fz])
        except:
            print(i)
    return result

def filter_force(force, t0, t1):
    '''
    force: list of [timestamp, fx, fy, fz]
    t0, t1: timestamp of start and end time
    return: list of force within t0-t1
    '''
    result = [[] for i in range(3)]
    for item in force:
        time = item[0]
        if time > t0 and time < t1:
            result[0].append(item[1])
            result[1].append(item[2])
            result[2].append(item[3])
    return result


# Force data extraction
force_path = '/home/jc/logs/iros_video/log1.txt'
force_txt = load_txt(force_path)
force = extrat_force(force_txt)

# sort force based on action timestamp
t0 = force[0][0]

right_start = add_timestamp(t0, '01:23:25000')
right_end = add_timestamp(right_start, '00:10:3000')

left_start = add_timestamp(t0, '01:33:010000')
left_end = add_timestamp(left_start, '00:10:25000')

up_start = add_timestamp(t0, '01:39:7000')
up_end = add_timestamp(up_start, '00:10:6000')

# extrat force data for 3 actions 
force_right = filter_force(force, right_start, right_end)
force_left = filter_force(force, left_start, left_end)
force_up = filter_force(force, up_start, up_end)

# plot force and do animation
import matplotlib.pyplot as plt
plt.ion()
fig, axe = plt.subplots(1,1)
axe.set_xlim(0, len(force_right[0]))
for i in range(len(force_right[0])):
    axe.plot(force_right[0][:i], 'r')
    axe.plot(force_right[1][:i], 'g')
    axe.plot(force_right[2][:i], 'b')
    plt.pause(0.1)