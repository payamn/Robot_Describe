#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

from move_base_msgs.msg import MoveBaseActionGoal
# from numpy.doc.constants import prev
from sensor_msgs.msg import LaserScan

from std_msgs.msg import String
import random
from tf import TransformListener
import numpy as np
import threading
import rospkg
import os
from utility import *
import pickle
import math
from utils.configs import *

is_write = True

tf_listner = TransformListener()
idx = 0
counter = 0
positions = []
def callback_robot_0(odom_data):
    global tf_listner, index_prev_points, prev_points, positions, idx, counter
    # if not tf_listner.frameExists("/robot_0/base_link"):
    #     print "/robot_0/base_link not exist"
    # if not tf_listner.frameExists("map"):
    #     print "map dont exits"
    # if tf_listner.frameExists("/robot_0/base_link") and tf_listner.frameExists("/map"):
    t = tf_listner.getLatestCommonTime("/robot_0/base_link", "/map")
    position, quaternion = tf_listner.lookupTransform("/map", "/robot_0/base_link", t)
    # print position
    positions.append(position)
    counter += 1
    if counter % 100 == 0:
        pickle.dump(positions,
                    open(rospkg.RosPack().get_path('robot_describe') + "/script/data/{}/{}.p".format(MAP_NAME, idx), "wb"))
        print "saving", idx
        print positions
        positions = []
        counter = 0
        idx += 1

def write_to_pickle():
    global points_description, index_senteces
    rospy.init_node('Path_saver')
    rospy.Subscriber("/robot_0/base_pose_ground_truth", Odometry, callback_robot_0)

    rospy.spin()
def read_from_pickle():
    rospy.spin()


if __name__ == '__main__':
    if is_write:
        if not os.path.isdir(rospkg.RosPack().get_path('robot_describe') + "/script/data/{}/".format(MAP_NAME)):
            os.makedirs(rospkg.RosPack().get_path('robot_describe') + "/script/data/{}/".format(MAP_NAME))
        write_to_pickle()
    else:
        read_from_pickle()

