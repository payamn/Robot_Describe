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

is_write = False
use_full_sentences = ["room on right side", "room on left side","T junction", "Corner, Left Turn", "Corner, Right turn", "Long Corridor", "Narrow Corridor", "Turn Left or Right", "Go Forward or Right", "Go left or Forward" ]
index_senteces = 0
points_description = []

from robot import *
tf_listner = None
diff_to_save_laser = 8
prev_points = [[0,0] for x in range (5)]
index_prev_points = 0

def callback_laser_scan(scan, my_robot):
    my_robot.save_bag_scan_laser(scan)


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
            degree_diff = (degree_robot[2] - nodes[1][index][0]) * math.pi / 180
            degree_diff = arctan2(sin(degree_diff), cos(degree_diff)) * 180 / math.pi
            # print degree_diff
            if (degree_diff<40 and degree_diff>-40):
                list_return_index.append(index)
            # print degree_robot, no  des[1][index]
    # print "min",min_dist
    # print ""
    return list_return_index

# def callback_robot_0(odom_data, my_robot):
#     pose = odom_data.pose.pose.position
#     my_robot.pose = array([pose.x+mapper_constant_val_x, pose.y + mapper_constant_val_y])
#     my_robot.check_next_intersection()
def callback_robot_0(odom_data, my_robot):
    global tf_listner, index_prev_points, prev_points
    # if not tf_listner.frameExists("/robot_0/base_link"):
    #     print "/robot_0/base_link not exist"
    # if not tf_listner.frameExists("map"):
    #     print "map dont exits"
    # if tf_listner.frameExists("/robot_0/base_link") and tf_listner.frameExists("/map"):
    t = tf_listner.getLatestCommonTime("/robot_0/base_link", "/map")
    position, quaternion = tf_listner.lookupTransform("/map", "/robot_0/base_link", t)
    # print position

    if (odom_data.twist.twist.linear.x == 0 and odom_data.twist.twist.linear.y == 0 and odom_data.twist.twist.angular.z ==0):
        my_robot.set_stop(True)
        return
    else:
        my_robot.set_stop(False)
        my_robot.set_speed(odom_data.twist.twist)


    pose = position
    robot_orientation_degree = quaternion_to_euler_angle(odom_data.pose.pose.orientation.w, odom_data.pose.pose.orientation.x, odom_data.pose.pose.orientation.y,
                              odom_data.pose.pose.orientation.z)
    list_index =  closest_node(np.asarray([pose[0], pose[1]]), points_description, robot_orientation_degree, 1.4)
    for index in list_index:
        happend_recently = False
        for point in prev_points:
            if point[0] == points_description[0][index][0] and point[1] == points_description[0][index][1]:
                happend_recently = True
                break
        if happend_recently:
            continue
        prev_points[index_prev_points] =  points_description[0][index]
        index_prev_points = (index_prev_points + 1)%5

        # print points_description[0][index]
        # print (points_description[1][index][1])
        my_robot.add_lang_str(points_description[1][index][1])

def quaternion_to_euler_angle(w, x, y, z):
    ysqr = y * y

    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + ysqr)
    X = math.degrees(math.atan2(t0, t1))

    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    Y = math.degrees(math.asin(t2))

    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (ysqr + z * z)
    Z = math.degrees(math.atan2(t3, t4))

    return X, Y, Z

def callback(data):
    global index_senteces,use_full_sentences
    _, _, z = quaternion_to_euler_angle(data.pose.orientation.w, data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z)
    point = [data.pose.position, z, use_full_sentences[index_senteces]]
    points_description.append(point)
    print('one Point for %s saved'%points_description[-1][2])

def write_to_pickle():
    global points_description, index_senteces
    rospy.init_node('point_saver')
    rospy.Subscriber("/move_base_simple/goal", PoseStamped, callback)
    for i in range(len(use_full_sentences)):
        index_senteces = i
        print use_full_sentences[i]
        cmd = None
        while cmd!="":
            cmd = raw_input("when finished enter or use r to remove last point:\n")
            if cmd == "r":
                points_description = points_description[0:-1]
            if cmd == "len":
                print len(points_description)
            if cmd == "show":
                print points_description

    pickle.dump(points_description,
                open(rospkg.RosPack().get_path('robot_describe') + "/script/data/points_describtion.p", "wb"))

    rospy.spin()
def read_from_pickle():
    global points_description, tf_listner
    rospy.init_node('point_test')
    tf_listner = TransformListener()
    points_description = pickle.load(open(rospkg.RosPack().get_path('robot_describe') + "/script/data/points_describtion.p", "rb"))
    position_points = [[i[0].x, i[0].y] for i in points_description]
    description_degree = [[i[1], i[2]] for i in points_description]
    # for points in description_degree:
    #     print points
    points_description = [position_points, description_degree]
    print description_degree
    robot = Robot(True, 2, start_index_dataset=1140, use_direction_file=False, bag_name="/bag_train/")
    rospy.Subscriber("/robot_0/base_pose_ground_truth", Odometry, callback_robot_0, robot)
    rospy.Subscriber("/robot_0/base_scan_1", LaserScan, callback_laser_scan, robot)
    rospy.spin()


if __name__ == '__main__':
   if is_write:
       write_to_pickle()
   else:
       read_from_pickle()

