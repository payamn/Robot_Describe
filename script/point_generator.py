#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped

from move_base_msgs.msg import MoveBaseActionGoal
from sensor_msgs.msg import LaserScan

from std_msgs.msg import String
import random
import numpy as np
import threading
import rospkg
from utility import *
import pickle
import math

is_write = False
use_full_sentences = ["T junction", "Corner, Left Turn", "Corner, Right turn", "Long Corridor", "Narrow Corridor", "Turn Left or Right", "Go Forward or Right", "Go left or Forward" ]
index_senteces = 0
points_description = []

from robot import *

prev_saved_laser = []
diff_to_save_laser = 8
#
# def callback_laser_scan(scan, my_robot):
#     global prev_saved_laser
#     if (len(prev_saved_laser)>0):
#         diff = 0
#         for i in range (len(prev_saved_laser)):
#             diff += abs(prev_saved_laser[i] - scan.ranges[i])
#         if diff > diff_to_save_laser:
#             prev_saved_laser = scan.ranges
#             my_robot.data_set.write_bag("/robot_0/base_scan_1", scan)
#             print "saving" + str(diff)
#         else:
#             print "not saving" + str(diff)
#     else:
#         prev_saved_laser = scan.ranges
#         my_robot.data_set.write_bag("/robot_0/base_scan_1", scan)


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    dist_2 = np.sum((nodes - node)**2, axis=1)
    return np.argmin(dist_2)

# def callback_robot_0(odom_data, my_robot):
#     pose = odom_data.pose.pose.position
#     my_robot.pose = array([pose.x+mapper_constant_val_x, pose.y + mapper_constant_val_y])
#     my_robot.check_next_intersection()
def callback_robot_0(odom_data):
    pose = odom_data.pose.pose.position
    index =  closest_node(np.asarray([pose.x, pose.y]), points_description[0])
    print (points_description[0][index], points_description[1][index])

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
    _, _, z = quaternion_to_euler_angle(data.pose.orientation.w, data.pose.orientation.x, data.pose.orientation.y, data.pose.orientation.z)
    point = [data.pose.position, z, use_full_sentences[index_senteces]]
    points_description.append(point)
    print('one Point for %s saved'%use_full_sentences[i])

def write_to_pickle():
    rospy.init_node('point_saver')
    rospy.Subscriber("/move_base_simple/goal", PoseStamped, callback)
    for i in range(len(use_full_sentences)):
        index_senteces = i
        print use_full_sentences[i]
        wait = raw_input("when finished enter\n")
    pickle.dump(points_description,
                open(rospkg.RosPack().get_path('robot_describe') + "/script/data/points_describtion.p", "wb"))

    rospy.spin()
def read_from_pickle():
    global points_description
    rospy.init_node('point_test')

    points_description = pickle.load(open(rospkg.RosPack().get_path('robot_describe') + "/script/data/points_describtion.p", "rb"))
    position_points = [[i[0].x, i[0].y] for i in points_description]
    description_degree = [[i[1], i[2]] for i in points_description]
    points_description = [position_points, description_degree]
    print description_degree
    # robot = Robot(True, 1)
    rospy.Subscriber("/robot_0/base_pose_ground_truth", Odometry, callback_robot_0)
    # rospy.Subscriber("/robot_0/base_pose_ground_truth", Odometry, callback_robot_0, robot)
    # rospy.Subscriber("/robot_0/base_scan_1", LaserScan, callback_laser_scan, robot)
    rospy.spin()


if __name__ == '__main__':
   if is_write:
       write_to_pickle()
   else:
       read_from_pickle()

