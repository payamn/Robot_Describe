#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from move_base_msgs.msg import MoveBaseActionGoal
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import random
from numpy import *
import threading
import rospkg
from utility import *
from robot import *

prev_saved_laser = []
diff_to_save_laser = 8

def callback_laser_scan(scan, my_robot):
    global prev_saved_laser
    if (len(prev_saved_laser)>0):
        diff = 0
        for i in range (len(prev_saved_laser)):
            diff += abs(prev_saved_laser[i] - scan.ranges[i])
        if diff > diff_to_save_laser:
            prev_saved_laser = scan.ranges
            my_robot.data_set.write_bag("/robot_0/base_scan_1", scan)
            print "saving" + str(diff)
        else:
            print "not saving" + str(diff)
    else:
        prev_saved_laser = scan.ranges
        my_robot.data_set.write_bag("/robot_0/base_scan_1", scan)


def callback_robot_0(odom_data, my_robot):
    pose = odom_data.pose.pose.position
    my_robot.pose = array([pose.x+mapper_constant_val_x, pose.y + mapper_constant_val_y])
    my_robot.check_next_intersection()

if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    robot = Robot(True, 1)
    rospy.Subscriber("/robot_0/base_pose_ground_truth", Odometry, callback_robot_0, robot)
    rospy.Subscriber("/robot_0/base_scan_1", LaserScan, callback_laser_scan, robot)
    rospy.spin()
