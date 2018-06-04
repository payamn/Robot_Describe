#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry, OccupancyGrid
from nav_msgs.srv import GetMap
from geometry_msgs.msg import PoseStamped

from move_base_msgs.msg import MoveBaseActionGoal
# from numpy.doc.constants import prev
from sensor_msgs.msg import LaserScan

from stage_ros.srv import reset_position
from std_msgs.msg import String
import random
from geometry_msgs.msg import Pose
from tf import TransformListener, transformations
import numpy as np
import threading
import rospkg

import cv2

import os
import subprocess

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
LOCAL_DISTANCE = 4

idx = 0
counter = 0
map = {}
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


# rospy.wait_for_service('/dynamic_map')
def call_back_laser_read(odom_data):
    get_map()


def get_map():
    global  map
    try:
        gmap_service = rospy.ServiceProxy('/dynamic_map', GetMap)
        gmap = gmap_service()

        map_array = np.array(gmap.map.data).reshape((gmap.map.info.height, gmap.map.info.width))
        map_array = normalize(map_array)
        map["data"] = map_array
        map["info"] = gmap.map.info
        get_local_map()
        return map
    except rospy.ServiceException, e:
        print "Service call failed: %s" % e


def read_from_pickle(pickle_dir):
    rospy.init_node('Path_follower')
    rospy.Subscriber("/robot_0/base_pose_ground_truth", Odometry, call_back_laser_read)

    files = [f for f in os.listdir(pickle_dir) if os.path.isfile(os.path.join(pickle_dir, f))]
    files = sorted(files, key=lambda file: int(file.split('.')[0]))
    files = [os.path.join(pickle_dir, f) for f in files]

    publisher = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10, latch=True)
    coordinates = []
    for file in files:
        with open(file, "rb") as p:
            data = pickle.load(p)
            coordinates = coordinates + data

    random.seed(a=None)
    index = random.randint(0, len(coordinates)-1)
    index += 1
    move_robot_to_pose(coordinates[index])
    print "start index: ", index, " len: ", len(coordinates)
    time.sleep(0.1)
    counter = 10
    while (index < len(coordinates)):
        counter -= 1

        if counter <= 0:
            counter = 10
            reset_gmapping()
        position, quaternion = Utility.get_robot_pose("/map_server")

        if Utility.distance_vector(position[:2], coordinates[index][:2]) < MIN_DISTANCE_TO_PUBLISH:
            index += 1
            continue


        while Utility.distance_vector(position[:2], coordinates[index][:2]) > MAX_DISTANCE_TO_PUBLISH:
            time.sleep(0.3)
            position, quaternion = Utility.get_robot_pose("/map_server")
            # print ("distance greater waiting")

        target_goal_simple = PoseStamped()
        target_goal_simple.pose.position.x = coordinates[index][0]
        target_goal_simple.pose.position.y = coordinates[index][1]
        target_goal_simple.pose.position.z = 0
        target_goal_simple.pose.orientation.w = 1
        target_goal_simple.header.frame_id = 'map_server'
        target_goal_simple.header.stamp = rospy.Time.now()
        publisher.publish(target_goal_simple)
        index += 1

    print "finish execution"
    rospy.spin()

def reset_gmapping():
    p = subprocess.Popen("rosnode kill /slam_gmapping", stdout=None, shell=True)
    (output, err) = p.communicate()
    p.wait()
    p = subprocess.Popen("roslaunch robot_describe gmapping.launch", stdout=None, shell=True)
    time.sleep(0.5)
    print ("gmapping reset")

reset_gmapping()

def call_back_map(data):
    global map
    print (len(data.data))
    map_array = np.array(data.data).reshape((data.info.height, data.info.width))
    map_array = normalize(map_array)
    map = map_array, data.info.resolution
    get_local_map()
    return 0


def get_local_map():
    global map
    if len(map) == 0:
        return
    # get robot pose in gmapping
    position, quaternion = Utility.get_robot_pose("/map")
    plus_x = map["info"].origin.position.y * map["info"].resolution
    plus_y = map["info"].origin.position.x * map["info"].resolution
    position[0] -= map["info"].origin.position.x #* map["info"].resolution
    position[1] -= map["info"].origin.position.y #* map["info"].resolution
    _, _, z = Utility.quaternion_to_euler_angle(quaternion[0], quaternion[1],
                                                quaternion[2], quaternion[3])


    image = Utility.sub_image(map["data"], map["info"].resolution, (position[0]  , position[1]), z,
                      LOCAL_DISTANCE * 4, LOCAL_DISTANCE * 4, only_forward=True)

    cv2.imshow("local", image)
    cv2.waitKey(1)
    return image

def normalize(data):
    data -= data.min()
    data /= (data.max() - data.min())
    data *= 255
    data = data.astype(np.uint8)
    return data

if __name__ == '__main__':
    pickle_dir = rospkg.RosPack().get_path('robot_describe') + "/script/data/{}/".format(MAP_NAME)
    if is_write:
        if not os.path.isdir(pickle_dir):
            os.makedirs(pickle_dir)
        write_to_pickle()
    else:
        read_from_pickle(pickle_dir)

