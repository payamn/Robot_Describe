#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped

from move_base_msgs.msg import MoveBaseActionGoal
# from numpy.doc.constants import prev
from sensor_msgs.msg import LaserScan
import cv2

from std_msgs.msg import String
import random
import tf
from tf import TransformListener
import numpy as np
import threading
import rospkg
from utility import *
import pickle
import math
from utils import model_utils
from utils.configs import *

class GenerateMap:
    def __init__(self, start_pickle = 0):
        points = []
        rospy.init_node('GenerateMap')
        self.prev_points = [([0, 0],0 )for x in range(10)]
        self.index_prev_points = 0
        self.points_description = []
        self.tf_listner = TransformListener()
        self.word_encoding = model_utils.WordEncoding()
        self.sentences = self.word_encoding.sentences
        self.classes = self.word_encoding.classes
        self.sentences_index = 0
        self.max_distace = 3
        self.pose = (10, 10, 0)
        self.robot_orientation_degree = (0,0,0)
        self.map = None
        self.language = None
        self.data_pointer = 0
        self.laser_list = []
        self.speed_list = []
        self.local_map_list = []
        self.current_speed = None
        self.len_to_save = 20
        self.pickle_counter = start_pickle
        self.stop = True
        self.skip_couter = 2
        self.skip_len = 2

    def save_pickle(self):
        lasers = []
        speeds = []
        local_maps = []
        for index in range(self.len_to_save):
            lasers = self.laser_list[self.data_pointer:] + self.laser_list[0:self.data_pointer]
            speeds = self.speed_list[self.data_pointer:] + self.speed_list[0:self.data_pointer]
            local_maps = self.local_map_list[self.data_pointer:] + self.local_map_list[0:self.data_pointer]
        print ("saving {}.pkl language:{}".format(self.pickle_counter, self.language))
        data = {"laser_scans":lasers, "speeds":speeds, "local_maps":local_maps, "language":self.language}
        pickle.dump(data,
                    open(rospkg.RosPack().get_path('robot_describe') + "/script/data/dataset/train/{}_{}.pkl".format(MAP_NAME, self.pickle_counter), "wb"))
        self.pickle_counter += 1

    def add_data(self, laser, local_map): # add a data to local_map list
        if self.stop:
            return

        if len(self.speed_list) != self.len_to_save:
            self.speed_list.append(self.current_speed)
            self.laser_list.append(laser)
            self.local_map_list.append(local_map)
        else:
            self.speed_list[self.data_pointer] = self.current_speed
            self.laser_list[self.data_pointer] = laser
            self.local_map_list[self.data_pointer] = local_map
            self.data_pointer = (self.data_pointer + 1) % self.len_to_save
            self.skip_couter -= 1
            if (self.skip_couter < 0):
                self.save_pickle()
                self.skip_couter = self.skip_len

    def callback_point(self, data):
        _, _, z = Utility.quaternion_to_euler_angle(data.pose.orientation.w, data.pose.orientation.x,
                                                    data.pose.orientation.y, data.pose.orientation.z)
        point = [data.pose.position, z, self.sentences[self.sentences_index]]
        self.points_description.append(point)
        print('one Point for %s saved' % self.points_description[-1][2])

    def get_local_map(self):
        image = Utility.sub_image(self.map[0], self.map[1], (self.pose[0], self.pose[1]), self.robot_orientation_degree[2],
                          self.max_distace * 4, self.max_distace * 4)
        cv2.imshow("local", image)
        cv2.waitKey(1)
        return image

    def callback_laser_scan(self, scan):
        if self.map is None or self.current_speed is None or self.language is None:
            return
        local_map = self.get_local_map()
        laser_data = [float(x) / scan.range_max for x in scan.ranges]
        self.add_data(laser_data, local_map)

    def callback_robot_0(self, odom_data):

        t = self.tf_listner.getLatestCommonTime("/robot_0/base_link", "/map")
        position, quaternion = self.tf_listner.lookupTransform("/map", "/robot_0/base_link", t)
        if (
                odom_data.twist.twist.linear.x == 0 and
                odom_data.twist.twist.linear.y == 0 and
                odom_data.twist.twist.angular.z == 0
            ):
            self.stop = True
            return
        else:
            self.stop = False
            self.current_speed = \
                (odom_data.twist.twist.linear.x, odom_data.twist.twist.linear.y, odom_data.twist.twist.angular.z)

        self.pose = position
        self.robot_orientation_degree = Utility.quaternion_to_euler_angle(odom_data.pose.pose.orientation.w,
                                                                     odom_data.pose.pose.orientation.x,
                                                                     odom_data.pose.pose.orientation.y,
                                                                     odom_data.pose.pose.orientation.z)

        return_nodes = closest_node(np.asarray([self.pose[0], self.pose[1]]), self.points_description, self.robot_orientation_degree, self.max_distace
                                  , self.tf_listner)
        for index, nodes in enumerate(return_nodes):
            happend_recently = False
            for point in self.prev_points:
                if point[0][0] == self.points_description[0][nodes[0]][0] and point[0][1] == self.points_description[0][nodes[0]][1]:
                    happend_recently = True
                    break
            if happend_recently:
                return_nodes[index] = (nodes[0], point[1], nodes[2])
            self.prev_points[self.index_prev_points] = (self.points_description[0][return_nodes[index][0]], return_nodes[index][1])
            self.index_prev_points = (self.index_prev_points + 1) % 10
        self.language = return_nodes


    def read_from_pickle(self):
        self.points_description = pickle.load(
            open(rospkg.RosPack().get_path('robot_describe') + "/script/data/{}.p".format(MAP_NAME), "rb"))
        position_points = [[i[0].x, i[0].y] for i in self.points_description]
        description_degree = [[i[1], i[2]] for i in self.points_description]
        # for points in description_degree:
        self.points_description = [position_points, description_degree]
        print description_degree
        rospy.Subscriber("/robot_0/base_pose_ground_truth", Odometry, self.callback_robot_0)
        rospy.Subscriber("/robot_0/base_scan_1", LaserScan, self.callback_laser_scan)
        rospy.Subscriber("/map", OccupancyGrid, self.call_back_map)
        rospy.spin()

    def append_to_pickle(self):
        self.points_description = pickle.load(
            open(rospkg.RosPack().get_path('robot_describe') + "/script/data/{}.p".format(MAP_NAME), "rb"))
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
                    open(rospkg.RosPack().get_path('robot_describe') + "/script/data/{}.p".format(MAP_NAME), "wb"))



    def call_back_map(self, data):
        print (len(data.data))
        map_array = np.array(data.data).reshape((data.info.height, data.info.width))
        map_array = Utility.normalize(map_array)
        self.map = map_array, data.info.resolution



def degree_to_object(degree_object, degree_robot):
    degree_diff = (degree_robot - degree_object) * math.pi / 180
    degree_diff = np.arctan2(math.sin(degree_diff), math.cos(degree_diff)) * 180 / math.pi
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
        # rospy.logerr("unknown intersection")
        class_prediction = "t_junction"

    return class_prediction

def corner(degree):
    if degree < 0:
        class_prediction = "corner_right"
    else:
        class_prediction = "corner_left"

    return class_prediction


lang_dic = {"room":room, "t_junction":t_junction, "corner":corner}

def make_pose_stamped(pose, degree):
    out = PoseStamped()
    out.header.frame_id = "/map"
    out.header.stamp = rospy.Time(0)
    out.pose.position.z = 0
    out.pose.position.x = pose[0]
    out.pose.position.y = pose[1]
    out.pose.orientation.w = degree
    return out

def closest_node(robot_pos, nodes, degree_robot, max_r, tf_listner):
    nodes_pos = np.asarray(nodes[0])
    list_return = []
    min_dist = 100
    for index,point in enumerate(nodes_pos):
        dist = np.linalg.norm(point - robot_pos)
        min_dist = min(min_dist, dist)

        if (dist < max_r):
            degree = degree_to_object(nodes[1][index][0], degree_robot[2])
            pose = make_pose_stamped(point, nodes[1][index][0])
            pose2 = tf_listner.transformPose("/robot_0/base_link", pose)
            list_return.append((index, lang_dic[nodes[1][index][1]](degree), (pose2.pose.position.x, pose2.pose.position.y)))

    return list_return


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='generate map')
    parser.add_argument('--generate_point', dest='generate_point', action='store_true')
    # parser.add_argument('--foo', type=int, default=42, help='FOO!')
    parser.set_defaults(generate_point=False)
    args = parser.parse_args()

    generate_map = GenerateMap(start_pickle=0)

    if args.generate_point:
        generate_map.write_to_pickle()
    else:
        generate_map.read_from_pickle()

