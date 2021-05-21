# Python 2 scripts for real robot data collection
import os
import serial
import argparse
import syslog
import time
import datetime as dt


def main(args):
    port = '/dev/ttyACM0'

    ard = serial.Serial(port,9600,timeout=5)

    parent = os.path.dirname(args.output_path)
    if not os.path.exists(parent):
    	os.makedirs(parent)

    if args.new:
    	f = open(args.output_path, 'w')
    else:
	    f = open(args.output_path, 'a')
    i = 0
    while True:

        # Serial read section
        msg = ard.readline()
        print(msg)
        now = dt.datetime.now()

        # message might be empty, we will not log empty msg
        if i > 20 and msg.strip():
            f.write('time: {}/{} {}:{}:{},{}  force:  '
                    .format(now.month, now.day, now.hour, now.minute, now.second, now.microsecond))
            f.write(msg)
        # time.sleep(0.05)
        i += 1
    else:
        print "Exiting"
    exit()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Record tri-axis force sensor data from arduino')
    parser.add_argument('--t', default=300, type=float, help='time of seconds to record the data')
    parser.add_argument('--output_path', default='/home/jc/logs/realrobot/force_1.txt', help='output file path')
    parser.add_argument('--new', action='store_true', help="start new file instead of append to existing ones")
    args = parser.parse_args()
    
    main(args)