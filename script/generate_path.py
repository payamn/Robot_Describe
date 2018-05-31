#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

from move_base_msgs.msg import MoveBaseActionGoal
# from numpy.doc.constants import prev
from sensor_msgs.msg import LaserScan

from stage_ros.srv import reset_position
from std_msgs.msg import String
import random
from geometry_msgs.msg import Pose
from tf import TransformListener
import numpy as np
import threading
import rospkg
import os
from utility import Utility
import pickle
import math
from utils.configs import *
import time

from pid import PID

is_write = False

tf_listner = TransformListener()
MIN_DISTANCE_TO_PUBLISH = 5
MAX_DISTANCE_TO_PUBLISH = 7
idx = 0
counter = 0
positions = []
reset_pos_robot = rospy.ServiceProxy('/reset_position_robot_0', reset_position)

def callback_robot_0(odom_data):
    global tf_listner, index_prev_points, prev_points, positions, idx, counter
    # if not tf_listner.frameExists("/robot_0/base_link"):
    #     print "/robot_0/base_link not exist"
    # if not tf_listner.frameExists("map"):
    #     print "map dont exits"
    # if tf_listner.frameExists("/robot_0/base_link") and tf_listner.frameExists("/map"):
    if odom_data.twist.twist.linear.x == 0 and odom_data.twist.twist.linear.y == 0 and odom_data.twist.twist.angular.z == 0:
        print ("robot not moving skiping the sequence")
        return

    t = tf_listner.getLatestCommonTime("/robot_0/base_link", "/map_server")
    position, quaternion = tf_listner.lookupTransform("/map_server", "/robot_0/base_link", t)
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

def move_robot_to_pose(coordinate):
    pose = Pose()

    pose.position.x = coordinate[0] - 30.15
    pose.position.y = coordinate[1] - 26.5
    pose.position.z = 0
    pose.orientation.w = coordinate[2]

    reset_pos_robot(pose)

def write_to_pickle():
    global points_description, index_senteces
    rospy.init_node('Path_saver')
    rospy.Subscriber("/robot_0/base_pose_ground_truth", Odometry, callback_robot_0)

    rospy.spin()
def read_from_pickle(pickle_dir):
    rospy.init_node('Path_follower')

    files = [f for f in os.listdir(pickle_dir) if os.path.isfile(os.path.join(pickle_dir, f))]
    files = sorted(files, key=lambda file: int(file.split('.')[0]))
    files = [os.path.join(pickle_dir, f) for f in files]

    moved = False
    publisher = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10, latch=True)

    for file in files:
        with open(file, "rb") as p:
            data = pickle.load(p)
            for coordinate in data:
                if not moved:
                    move_robot_to_pose(coordinate)
                    moved = True
                    continue
                position, quaternion = Utility.get_robot_pose()

                if Utility.distance_vector(position[:2], coordinate[:2]) < MIN_DISTANCE_TO_PUBLISH:
                    continue

                while Utility.distance_vector(position[:2], coordinate[:2]) > MAX_DISTANCE_TO_PUBLISH:
                    time.sleep(0.3)
                    position, quaternion = Utility.get_robot_pose()

                target_goal_simple = PoseStamped()
                target_goal_simple.pose.position.x = coordinate[0]
                target_goal_simple.pose.position.y = coordinate[1]
                target_goal_simple.pose.position.z = 0
                target_goal_simple.pose.orientation.w = 1
                target_goal_simple.header.frame_id = 'map_server'
                target_goal_simple.header.stamp = rospy.Time.now()
                publisher.publish(target_goal_simple)
                print coordinate

    rospy.spin()


if __name__ == '__main__':
    pickle_dir = rospkg.RosPack().get_path('robot_describe') + "/script/data/{}/".format(MAP_NAME)
    if is_write:
        if not os.path.isdir(pickle_dir):
            os.makedirs(pickle_dir)
        write_to_pickle()
    else:
        read_from_pickle(pickle_dir)

