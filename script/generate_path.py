#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry, OccupancyGrid
from nav_msgs.srv import GetMap
from geometry_msgs.msg import PoseStamped

from move_base_msgs.msg import MoveBaseActionGoal
# from numpy.doc.constants import prev
from sensor_msgs.msg import LaserScan

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


class GeneratePath:
    def __init__(self, pickle_dir, map_name = None, mode = None):
        if map_name is None or mode is None:
            self.map_name = MAP_NAME
            self.mode = MODE
        self.MAP_NAME = map_name
        self.tf_listner = TransformListener()
        self.MIN_DISTANCE_TO_PUBLISH = 5
        self.MAX_DISTANCE_TO_PUBLISH = 7
        self.LOCAL_DISTANCE = 5.6

        self.idx = 0
        self.counter = 0
        self.map = {}
        self.positions = []
        if not os.path.isdir(pickle_dir):
            os.makedirs(pickle_dir)
        files = [f for f in os.listdir(pickle_dir) if os.path.isfile(os.path.join(pickle_dir, f))]
        files = sorted(files, key=lambda file: int(file.split('.')[0]))
        files = [os.path.join(pickle_dir, f) for f in files]

        self.publisher = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10, latch=True)
        self.coordinates = []
        self.base_frame = rospy.get_param("/base_frame")

        for file in files:
            with open(file, "rb") as p:
                data = pickle.load(p)
                self.coordinates = self.coordinates + data

    def callback_robot_0(self, odom_data):
        if odom_data.twist.twist.linear.x == 0 and odom_data.twist.twist.linear.y == 0 and odom_data.twist.twist.angular.z == 0:
            print ("robot not moving skiping the sequence")
            return
        t = self.tf_listner.getLatestCommonTime(self.base_frame, "/map_server")
        position, quaternion = self.tf_listner.lookupTransform("/map_server", self.base_frame, t)
        # print position
        self.positions.append(position)
        self.counter += 1
        if self.counter % 100 == 0:
            pickle.dump(self.positions,
                        open(rospkg.RosPack().get_path('robot_describe') + "/script/data/{}_{}/{}.p".format(self.map_name, self.mode, self.idx), "wb"))
            print "saving", self.idx
            print self.positions
            self.positions = []
            self.counter = 0
            self.idx += 1



    def write_to_pickle(self):
        rospy.init_node('Path_saver')
        rospy.Subscriber("/base_pose_ground_truth", Odometry, self.callback_robot_0)

        rospy.spin()


    # rospy.wait_for_service('/dynamic_map')
    def call_back_laser_read(self, odom_data):
        self.get_map()


    def get_map(self):
        try:
            gmap_service = rospy.ServiceProxy('/dynamic_map', GetMap)
            gmap = gmap_service()

            map_array = np.array(gmap.map.data).reshape((gmap.map.info.height, gmap.map.info.width))
            map_array = Utility.normalize(map_array)
            self.map["data"] = map_array
            self.map["info"] = gmap.map.info
            self.get_local_map()
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    def get_coordinates(self):
        return self.coordinates

    def read_from_pickle(self):
        rospy.init_node('Path_follower')
        rospy.Subscriber("/base_pose_ground_truth", Odometry, self.call_back_laser_read)
        while True:
            random.seed(a=None)
            index = random.randint(0, len(self.coordinates)-1)
            index += 1
            self.move_robot_to_pose(self.coordinates[index])
            print "start index: ", index, " len: ", len(self.coordinates)
            time.sleep(0.1)
            counter = 30
            while (index < len(self.coordinates)):
                # counter -= 1
                #
                # if counter <= 0:
                #     counter = 30
                #     self.reset_gmapping()
                position, quaternion = Utility.get_robot_pose("/map_server")

                if Utility.distance_vector(position[:2], self.coordinates[index][:2]) < self.MIN_DISTANCE_TO_PUBLISH:
                    index += 1
                    continue


                while Utility.distance_vector(position[:2], self.coordinates[index][:2]) > self.MAX_DISTANCE_TO_PUBLISH:
                    time.sleep(0.3)
                    position, quaternion = Utility.get_robot_pose("/map_server")
                    # print ("distance greater waiting")

                target_goal_simple = PoseStamped()
                target_goal_simple.pose.position.x = self.coordinates[index][0]
                target_goal_simple.pose.position.y = self.coordinates[index][1]
                target_goal_simple.pose.position.z = 0
                target_goal_simple.pose.orientation.w = 1
                target_goal_simple.header.frame_id = 'map_server'
                target_goal_simple.header.stamp = rospy.Time.now()
                publisher.publish(target_goal_simple)
                index += 1

        print "finish execution"
        rospy.spin()




    # def call_back_map(self, data):
        # print (len(data.data))
        # map_array = np.array(data.data).reshape((data.info.height, data.info.width))
        # map_array = Utility.normalize(map_array)
        # self.map = map_array, data.info.resolution
        # self.get_local_map()


if __name__ == '__main__':
    # if True, robot should move in map to generate points then by changing it to false it will move the robot along the path
    is_generate_path = True

    pickle_dir = rospkg.RosPack().get_path('robot_describe') + "/script/data/{}_{}/".format(MAP_NAME, MODE)

    generate_path = GeneratePath(pickle_dir)
    if is_generate_path:
        if not os.path.isdir(pickle_dir):
            os.makedirs(pickle_dir)
        generate_path.write_to_pickle()
    else:
        generate_path.read_from_pickle()

