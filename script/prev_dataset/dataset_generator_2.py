#!/usr/bin/env python
import script.rospy
from nav_msgs.msg import Odometry
from move_base_msgs.msg import MoveBaseActionGoal
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import random
from numpy import *
import threading
import rospkg
from script.utility import *
from script.robot import *

def callback_laser_scan(scan, my_robot):
    my_robot.save_bag_scan_laser(scan)

def callback_robot_0(odom_data, my_robot):
    pose = odom_data.pose.pose.position
    my_robot.pose = array([pose.x+mapper_constant_val_x, pose.y + mapper_constant_val_y])
    my_robot.check_next_intersection()

if __name__ == '__main__':
    script.rospy.init_node('listener', anonymous=True)
    robot = Robot(True, 2, start_index_dataset=0)
    script.rospy.Subscriber("/robot_0/base_pose_ground_truth", Odometry, callback_robot_0, robot)
    script.rospy.Subscriber("/robot_0/base_scan_1", LaserScan, callback_laser_scan, robot)
    script.rospy.spin()