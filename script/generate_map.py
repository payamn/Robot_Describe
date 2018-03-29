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
from utility import *
import pickle
import math
from robot import *


class GenerateMap:
    def __init__(self):
        points = []
        rospy.init_node('GenerateMap')
        self.prev_points = [[0, 0] for x in range(5)]
        self.index_prev_points = 0
        self.points_description = []
        self.tf_listner = TransformListener()
        self.sentences = ["room", "T junction", "Corner"]
        self.sentences_index = 0

    def callback_point(self, data):
        _, _, z = Utility.quaternion_to_euler_angle(data.pose.orientation.w, data.pose.orientation.x,
                                                    data.pose.orientation.y, data.pose.orientation.z)
        point = [data.pose.position, z, self.sentences[self.sentences_index]]
        self.points_description.append(point)
        print('one Point for %s saved' % self.points_description[-1][2])

    def callback_laser_scan(self, scan, my_robot):
        my_robot.save_bag_scan_laser(scan)

    def callback_robot_0(self, odom_data, my_robot):
        t = self.tf_listner.getLatestCommonTime("/robot_0/base_link", "/map")
        position, quaternion = self.tf_listner.lookupTransform("/map", "/robot_0/base_link", t)
        # print position

        #debug commented
        # if (
        #         odom_data.twist.twist.linear.x == 0 and
        #         odom_data.twist.twist.linear.y == 0 and
        #         odom_data.twist.twist.angular.z == 0
        #     ):
        #     my_robot.set_stop(True)
        #     return
        # else:
        #     my_robot.set_stop(False)
        #     my_robot.set_speed(odom_data.twist.twist)

        pose = position
        robot_orientation_degree = Utility.quaternion_to_euler_angle(odom_data.pose.pose.orientation.w,
                                                                     odom_data.pose.pose.orientation.x,
                                                                     odom_data.pose.pose.orientation.y,
                                                                     odom_data.pose.pose.orientation.z)

        list_index = closest_node(np.asarray([pose[0], pose[1]]), self.points_description, robot_orientation_degree, 1.5)
        for index in list_index:
            happend_recently = False
            for point in self.prev_points:
                if point[0] == self.points_description[0][index][0] and point[1] == self.points_description[0][index][1]:
                    happend_recently = True
                    break
            if happend_recently:
                continue
            self.prev_points[self.index_prev_points] = self.points_description[0][index]
            self.index_prev_points = (self.index_prev_points + 1) % 5

            # print self.points_description[0][index]
            # print (self.points_description[1][index][1])
            my_robot.add_lang_str(self.points_description[1][index][1])

    def read_from_pickle(self):
        self.points_description = pickle.load(
            open(rospkg.RosPack().get_path('robot_describe') + "/script/data/points_describtion.p", "rb"))
        position_points = [[i[0].x, i[0].y] for i in self.points_description]
        description_degree = [[i[1], i[2]] for i in self.points_description]
        # for points in description_degree:
        #     print points
        self.points_description = [position_points, description_degree]
        print description_degree
        robot = Robot(True, 2, start_index_dataset=0, use_direction_file=False, bag_name="/bag_train/")
        rospy.Subscriber("/robot_0/base_pose_ground_truth", Odometry, self.callback_robot_0, robot)
        rospy.Subscriber("/robot_0/base_scan_1", LaserScan, self.callback_laser_scan, robot)
        rospy.spin()

    def append_to_pickle(self):
        self.points_description = pickle.load(
            open(rospkg.RosPack().get_path('robot_describe') + "/script/data/points_describtion.p", "rb"))
        self.write_to_pickle()

    def write_to_pickle(self):
        rospy.Subscriber("/move_base_simple/goal", PoseStamped, self.callback_point)
        for i in range(len(self.sentences)):
            self.sentences_index = i
            print self.sentences[i]
            cmd = None
            while cmd != "":
                cmd = raw_input("when finished enter or use r to remove last point:\n")
                if cmd == "r":
                    self.points_description = self.points_description[0:-1]
                if cmd == "len":
                    print len(self.points_description)
                if cmd == "show":
                    print self.points_description

        pickle.dump(self.points_description,
                    open(rospkg.RosPack().get_path('robot_describe') + "/script/data/points_describtion.p", "wb"))

def degree_to_object(degree_object, degree_robot):
    degree_diff = (degree_robot - degree_object) * math.pi / 180
    degree_diff = arctan2(sin(degree_diff), cos(degree_diff)) * 180 / math.pi
    return degree_diff

def room(degree):
    if degree<0:
        class_prediction = "room_right"
    else:
        class_prediction = "room_left"

    return class_prediction

def t_junction(degree):
    if -130<degree<-50:
        class_prediction = "t_junction_right_forward"
    elif -40<degree<40:
        class_prediction = "t_junction_right_left"
    elif 50<degree<130:
        class_prediction = "t_junction_left_forward"
    else:
        rospy.logerr("unknown intersection")
        class_prediction = "t_junction"

    return class_prediction

def corner(degree):
    if degree < 0:
        class_prediction = "corner_right"
    else:
        class_prediction = "corner_left"

    return class_prediction


lang_dic = {"room":room, "T junction":t_junction, "Corner":corner}


def closest_node(robot_pos, nodes, degree_robot, max_r):
    nodes_pos = np.asarray(nodes[0])
    list_return_index = []
    min_dist = 100
    for index,point in enumerate(nodes_pos):
        dist = np.linalg.norm(point - robot_pos)
        # print dist
        min_dist = min(min_dist, dist)
        # print degree_robot, math.pi
        # degree = degree_robot[2] * math.pi / 180
        # print degree_robot[2], arctan2(sin(degree), cos(degree))* 180 / math.pi

        if (dist < max_r):
            degree = degree_to_object(nodes[1][index][0], degree_robot[2])
            print(lang_dic[nodes[1][index][1]](degree), dist)

            # print degree_diff
            # if (degree_diff<40 and degree_diff>-40):
            #     list_return_index.append(index)
    return list_return_index





if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='generate map')
    parser.add_argument('--generate_point', dest='generate_point', action='store_true')
    # parser.add_argument('--foo', type=int, default=42, help='FOO!')
    parser.set_defaults(generate_point=False)
    args = parser.parse_args()

    generate_map = GenerateMap()

    if args.generate_point:
        generate_map.append_to_pickle()
    else:
        generate_map.read_from_pickle()

