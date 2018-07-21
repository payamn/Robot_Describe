#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry, OccupancyGrid
from nav_msgs.srv import GetMap

from geometry_msgs.msg import PoseStamped

from move_base_msgs.msg import MoveBaseActionGoal
# from numpy.doc.constants import prev
from sensor_msgs.msg import LaserScan, Image
import cv2
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

from generate_path import GeneratePath

from geometry_msgs.msg import Pose

from stage_ros.srv import reset_position

from std_msgs.msg import String
import random
import tf
from tf import TransformListener
import numpy as np

import os

import threading
import subprocess

import rospkg
from utility import *
import pickle
import math
import constants
from utils import model_utils
from utils.configs import *

class GenerateMap:
    def __init__(self, start_pickle = 0):
        self.is_init = False
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
        self.map = {}
        self.language = None
        self.data_pointer = 0
        self.image_pub = rospy.Publisher("local_map", Image, queue_size=1)
        self.image_pub_annotated = rospy.Publisher("local_map_annotated", Image, queue_size=1)
        self.laser_data = None
        self.laser_list = []
        self.speed_list = []
        self.local_map_list = []
        self.current_speed = None
        self.len_to_save = 20
        self.is_turning = 0
        self.pickle_counter = start_pickle
        self.stop = True
        self.skip_couter = 2
        self.skip_len = 2
        self.counter = 0
        self.local_map = None
        self.generate_path = ()
        self.reset_pos_robot = rospy.ServiceProxy('/reset_position_robot_0', reset_position)
        self.publisher = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10, latch=True)
        self.img_sub = None
        self.MIN_DISTANCE_TO_PUBLISH = 5
        self.MAX_DISTANCE_TO_PUBLISH = 7

        self.bridge = CvBridge()

        pickle_dir = rospkg.RosPack().get_path('robot_describe') + "/script/data/{}/".format(MAP_NAME)
        generate_path = GeneratePath(pickle_dir)
        self.coordinates = generate_path.get_coordinates()
        # self.img_sub = rospy.Subscriber("/cost_map_node/img", Image, self.callback_map_image,queue_size=1)

        # self.reset_gmapping()
        # self.get_map_info()

    def get_map_info(self):
        if not self.is_init:
            return
        # print "in get_get info"
        rospy.wait_for_service('/cost_map_node/map_info')
        try:
            map_info_service = rospy.ServiceProxy('/cost_map_node/map_info', GetMap)
            start_time = time.time()
            map_info = map_info_service()
            # print("get map --- %s seconds ---" % (time.time() - start_time))
            self.map["info"] = map_info.map.info
            # print "map_info set"
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    def get_map(self):
        rospy.wait_for_service('/dynamic_map')
        try:
            gmap_service = rospy.ServiceProxy('/dynamic_map', GetMap)
            start_time = time.time()
            gmap = gmap_service()
            # print("get map --- %s seconds ---" % (time.time() - start_time))
            map_array = np.array(gmap.map.data).reshape((gmap.map.info.height, gmap.map.info.width))
            map_array = Utility.normalize(map_array)
            self.map["data"] = map_array
            self.map["info"] = gmap.map.info
        except rospy.ServiceException, e:
            print "Service call failed: %s" % e

    # def reset_gmapping(self):


    import math

    def move_robot_to_pose_reset_gmap(self, coordinate, coordinate_next):
        self.is_init = False
        pose = Pose()

        angle = math.atan2(coordinate_next[1] - coordinate[1], coordinate_next[0] - coordinate[0])

        pose.position.x = coordinate[0] + X_OFFSET
        pose.position.y = coordinate[1] + Y_OFFSET
        pose.position.z = 0
        pose.orientation.w = angle

        self.reset_pos_robot(pose)
        # time.sleep(1)
        # cv2.destroyAllWindows()
        if self.img_sub:
            self.img_sub.unregister()
        p = subprocess.Popen("rosnode kill /cost_map_node", stdout=None, shell=True)
        (output, err) = p.communicate()
        p.wait()
        p = subprocess.Popen("rosnode kill /slam_gmapping", stdout=None, shell=True)
        (output, err) = p.communicate()
        p.wait()
        time.sleep(0.5)
        # print "angle", angle*180.0/math.pi

        self.laser_data = None
        self.laser_list = []
        self.speed_list = []
        self.local_map_list = []
        self.local_map = None
        self.current_speed = None
        self.language = None
        self.prev_points = [([0, 0],0 )for x in range(10)]

        time.sleep(0.5)

        p = subprocess.Popen("roslaunch robot_describe gmapping.launch", stdout=None, shell=True)
        rospy.wait_for_service('/cost_map_node/map_info')
        time.sleep(0.5)
        self.img_sub = rospy.Subscriber("/cost_map_node/img", Image, self.callback_map_image,queue_size=1)

        print ("gmapping reset")
        time.sleep(0.5)

        self.is_init = True

    def get_local_map(self, annotated_publish=True):
        self.get_map_info()
        # time.sleep(5)
        if not self.map.has_key("info") or not self.is_init:
            print "no map or not init"
            return
        # get robot pose in gmapping
        position, quaternion = Utility.get_robot_pose("/map")
        plus_x = self.map["info"].origin.position.y * self.map["info"].resolution
        plus_y = self.map["info"].origin.position.x * self.map["info"].resolution
        position[0] -= self.map["info"].origin.position.x #* self.map["info"].resolution
        position[1] -= self.map["info"].origin.position.y #* self.map["info"].resolution

        _, _, z = Utility.quaternion_to_euler_angle(quaternion[0], quaternion[1],
                                                    quaternion[2], quaternion[3])


        image = Utility.sub_image(self.map["data"], self.map["info"].resolution, (position[0]  , position[1]), z,
                          constants.LOCAL_MAP_DIM, constants.LOCAL_MAP_DIM, only_forward=True)
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(image))
        self.local_map = image.copy()

        if annotated_publish and self.language is not None:
            for visible in self.language:
                x = int(np.ceil(visible[2][0]/constants.LOCAL_MAP_DIM*image.shape[0] ))
                y = int(np.ceil((visible[2][1]/(constants.LOCAL_MAP_DIM/2.0) + 1)/2*image.shape[0]))

                print visible[1]
                cv2.circle(image, (x, y), 4, 100)

            self.image_pub_annotated.publish(self.bridge.cv2_to_imgmsg(image))




    def save_pickle(self):
        if not self.is_init or self.laser_data is None or self.local_map is None \
                or self.language is None or self.current_speed is None:
            return
        # lasers = []
        # speeds = []
        # local_maps = []
        # for index in range(self.len_to_save):
        #     lasers = self.laser_list[self.data_pointer:] + self.laser_list[0:self.data_pointer]
        #     speeds = self.speed_list[self.data_pointer:] + self.speed_list[0:self.data_pointer]
        #     local_maps = self.local_map_list[self.data_pointer:] + self.local_map_list[0:self.data_pointer]
        print ("saving {}.pkl language:{}".format(self.pickle_counter, self.language))
        data = {"laser_scan":self.laser_data, "speeds":self.current_speed, "local_maps":self.local_map, "language":self.language}
        pickle.dump(data,
                    open(rospkg.RosPack().get_path('robot_describe') + "/data/dataset/train/{}_{}.pkl".format(MAP_NAME, self.pickle_counter), "wb"))
        self.pickle_counter += 1

    # def add_data(self): # add a data to local_map list
    #     if self.stop or self.laser_data == None or self.local_map == None:
    #         return
    #
    #     if len(self.speed_list) != self.len_to_save:
    #         self.speed_list.append(self.current_speed)
    #         self.laser_list.append(self.laser_data)
    #         self.local_map_list.append(self.local_map)
    #     else:
    #         self.speed_list[self.data_pointer] = self.current_speed
    #         self.laser_list[self.data_pointer] = self.laser_data
    #         self.local_map_list[self.data_pointer] = self.local_map
    #         self.data_pointer = (self.data_pointer + 1) % self.len_to_save
    #
    #         self.skip_couter -= 1
    #         if (self.skip_couter < 0):
    #             self.save_pickle()
    #             self.skip_couter = self.skip_len

    def callback_point(self, data):
        _, _, z = Utility.quaternion_to_euler_angle(data.pose.orientation.x, data.pose.orientation.y,
                                                    data.pose.orientation.z, data.pose.orientation.w,)
        point = [data.pose.position, z, self.sentences[self.sentences_index]]
        self.points_description.append(point)
        print('one Point for %s saved' % self.points_description[-1][2])

    # def get_local_map(self):
    #     image = Utility.sub_image(self.map[0], self.map[1], (self.pose[0], self.pose[1]), self.robot_orientation_degree[2],
    #                       self.max_distace * 4, self.max_distace * 4)
    #     cv2.imshow("local", image)
    #     cv2.waitKey(1)
    #     return image

    def callback_laser_scan(self, scan):
        if not self.is_init:
            return
        # local_map = self.get_local_map()
        # if self.local_map is None or self.current_speed is None or self.language is None:
        #     return
        self.laser_data = [float(x) / scan.range_max for x in scan.ranges]
        # self.add_data(laser_data)

    def callback_map_image(self, image):
        if not self.is_init:
            return
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image, "mono8")
        except CvBridgeError as e:
            print(e)
            return
        if self.is_turning > 0:
            self.is_turning -= 1
            return
        map_array = Utility.normalize(cv_image)
        self.map["data"] = map_array
        # print ("local map updated")
        self.get_local_map()
        self.save_pickle()
        # print ("local map called")


    def callback_robot_0(self, odom_data):

        t = self.tf_listner.getLatestCommonTime("/base_link", "/map_server")
        position, quaternion = self.tf_listner.lookupTransform("/map_server", "/base_link", t)
        if (
                odom_data.twist.twist.linear.x == 0 and
                odom_data.twist.twist.linear.y == 0 and
                odom_data.twist.twist.angular.z == 0
            ):
            self.stop = True
            return
        else:
            if math.fabs(odom_data.twist.twist.angular.z) > 0.5:
                self.is_turning = 5
                # print "turning {}".format(odom_data.twist.twist.angular.z)
            elif math.fabs(odom_data.twist.twist.angular.z) > 0.3 and self.is_turning < 2:
                self.is_turning = 2


            self.stop = False
            self.current_speed = \
                (odom_data.twist.twist.linear.x, odom_data.twist.twist.linear.y, odom_data.twist.twist.angular.z)

        self.pose = position
        self.robot_orientation_degree = Utility.quaternion_to_euler_angle(
                                                                     odom_data.pose.pose.orientation.x,
                                                                     odom_data.pose.pose.orientation.y,
                                                                     odom_data.pose.pose.orientation.z,
                                                                     odom_data.pose.pose.orientation.w)
        if not self.is_init:
            return

        # new pose is half of the distance we want to cover for language topic
        pose_robot_start_of_image = [
            self.pose[0] + math.cos(self.robot_orientation_degree[2]*math.pi/180)*constants.LOCAL_DISTANCE / 2.0,
            self.pose[1] + math.sin(self.robot_orientation_degree[2]*math.pi/180)*constants.LOCAL_DISTANCE / 2.0,
            ]

        # closest node to the point half distance ahead of a robot so we will have objects that are ahead of robot
        return_nodes = closest_node(np.asarray([pose_robot_start_of_image[0],   pose_robot_start_of_image[1]]),
                                    self.points_description, self.robot_orientation_degree, constants.LOCAL_DISTANCE / 2.0
                                  , self.tf_listner)
        # for index, nodes in enumerate(return_nodes):

            # happend_recently = False
            # for point in self.prev_points:
            #     if point[0][0] == self.points_description[0][nodes[0]][0] and point[0][1] == self.points_description[0][nodes[0]][1]:
            #         happend_recently = True
            #         break
            # if happend_recently:
            #     return_nodes[index] = (nodes[0], point[1], nodes[2])

            # self.prev_points[self.index_prev_points] = (self.points_description[0][return_nodes[index][0]], return_nodes[index][1])
            # self.index_prev_points = (self.index_prev_points + 1) % 10

        self.language = return_nodes


    def read_from_pickle(self):
        while not self.is_init:
            time.sleep(0.1)

        self.points_description = pickle.load(
            open(rospkg.RosPack().get_path('robot_describe') + "/script/data/{}.p".format(MAP_NAME), "rb"))
        position_points = [[i[0].x, i[0].y] for i in self.points_description]
        description_degree = [[i[1], i[2]] for i in self.points_description]
        # for points in description_degree:
        self.points_description = [position_points, description_degree]
        # print description_degree
        rospy.Subscriber("/base_pose_ground_truth", Odometry, self.callback_robot_0, queue_size=1)
        rospy.Subscriber("/base_scan", LaserScan, self.callback_laser_scan, queue_size=1)
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

    def publish_point(self, coordinate, coordinate_next):
        target_goal_simple = PoseStamped()
        target_goal_simple.pose.position.x = coordinate[0]
        target_goal_simple.pose.position.y = coordinate[1]
        target_goal_simple.pose.position.z = 0
        w, x, y, z = Utility.toQuaternion(math.atan2(coordinate_next[1] - coordinate[1], coordinate_next[0] - coordinate[0]), 0, 0)
        target_goal_simple.pose.orientation.w = w
        target_goal_simple.pose.orientation.x = x
        target_goal_simple.pose.orientation.y = y
        target_goal_simple.pose.orientation.z = z


        target_goal_simple.header.frame_id = 'map_server'
        target_goal_simple.header.stamp = rospy.Time.now()
        self.publisher.publish(target_goal_simple)

    def point_generator(self):
        time.sleep(0.5)
        # we will first generate some short_range path and then long ones
        short_range = 50
        long_range = 20
        while True:
            random.seed(a=None)
            if short_range > 0:
                short_range -= 1
                start_index = random.randint(0, len(self.coordinates) - 100)
                end_index = min(random.randint(start_index + 50, start_index + 100)
                                , len(self.coordinates)-2)
                print ("{} short range remaining".format(short_range))
            elif long_range>0:
                long_range -= 1
                start_index = random.randint(0, len(self.coordinates) - 300)
                end_index = random.randint(start_index + 100, len(self.coordinates) - 2)
                print ("{} long range remaining".format(long_range))
            else:
                print ("finished all for this map")
                break

            self.publish_point(self.coordinates[start_index+10], self.coordinates[start_index+12])
            self.move_robot_to_pose_reset_gmap(self.coordinates[start_index], self.coordinates[start_index+2])
            self.publish_point(self.coordinates[start_index+10], self.coordinates[start_index+12])
            time.sleep(0.5)
            start_index += 14
            position, quaternion = Utility.get_robot_pose("/map_server")

            # while (Utility.distance_vector(position[:2], self.coordinates[start_index ][:2]) > self.MIN_DISTANCE_TO_PUBLISH):
            #     time.sleep(0.1)
            #     position, quaternion = Utility.get_robot_pose("/map_server")
            #     self.is_init = False

            print "start index: ", start_index, " len: ", end_index - start_index
            while not self.is_init:
                time.sleep(0.1)
            counter = 30
            while (start_index < end_index):
                # counter -= 1
                #
                # if counter <= 0:
                #     counter = 30
                #     self.reset_gmapping()
                position, quaternion = Utility.get_robot_pose("/map_server")

                if Utility.distance_vector(position[:2], self.coordinates[start_index][:2]) < self.MIN_DISTANCE_TO_PUBLISH:
                    start_index += 1
                    continue

                # for bug handling in case of navigation error
                counter_to_reset = 30
                while Utility.distance_vector(position[:2], self.coordinates[start_index][:2]) > self.MAX_DISTANCE_TO_PUBLISH\
                        and counter_to_reset > 0:
                    time.sleep(0.5)
                    self.publish_point(self.coordinates[start_index], self.coordinates[start_index+1])
                    position, quaternion = Utility.get_robot_pose("/map_server")
                    print ("distance greater waiting for {} seconds".format(counter_to_reset*0.5))
                    counter_to_reset -= 1
                self.publish_point(self.coordinates[start_index],self.coordinates[start_index+1])
                start_index += 1

                # if we are stuck
                if counter_to_reset <= 0:
                    # check if move base flex died
                    nodes = os.popen('rosnode list').read().split()
                    if "/move_base_flex" not in nodes:
                        # start move base flex node
                        p = subprocess.Popen("roslaunch robot_describe move_base_flex.launch", stdout=None, shell=True)
                        (output, err) = p.communicate()
                        p.wait()
                        print ("move base flex launch again because it was died unexpectedly")
                    break
            self.is_init = False
            time.sleep(0.1)
            # while (Utility.distance_vector(position[:2], self.coordinates[index-1][:2]) > 0.5):
            #     time.sleep(0.1)
            #     position, quaternion = Utility.get_robot_pose("/map_server")
        print "finish execution"

    # def call_back_map(self, data):
    #     print (len(data.data))
    #     map_array = np.array(data.data).reshape((data.info.height, data.info.width))
    #     map_array = Utility.normalize(map_array)
    #     self.map = map_array, data.info.resolution



def degree_to_object(degree_object, degree_robot):
    degree_diff = (degree_robot - degree_object) * math.pi / 180
    degree_diff = np.arctan2(math.sin(degree_diff), math.cos(degree_diff)) * 180 / math.pi
    return degree_diff

def close_room(degree):
    # if -150<degree<-30:
    #     class_prediction = "room_right"
    # elif 30<degree<150:
    #     class_prediction = "room_left"
    # else:
    #     class_prediction = None
    if -30<degree<30 or degree > 150 or degree < -150:
        class_prediction = None
    else:
        class_prediction = "close_room"
    return class_prediction

def open_room(degree):
    if -30 < degree < 30 or degree > 150 or degree < -150:
        class_prediction = None
    else:
        class_prediction = "open_room"
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
        class_prediction = None

    return class_prediction

def corner(degree):
    if degree < 0:
        class_prediction = "corner_right"
    else:
        class_prediction = "corner_left"

    return class_prediction

def junction(degree):
    return "4_junction"

lang_dic = {"close_room":close_room, "open_room":open_room, "t_junction":t_junction, "corner":corner, "4_junction":junction}

def make_pose_stamped(pose, degree):
    out = PoseStamped()
    out.header.frame_id = "/map_server"
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
            pose2 = tf_listner.transformPose("/base_link", pose)
            lang = lang_dic[nodes[1][index][1]](degree)
            if lang != None:
                list_return.append((index, lang_dic[nodes[1][index][1]](degree), (pose2.pose.position.x, pose2.pose.position.y)))

    return list_return


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='generate map')
    parser.add_argument('--generate_point', dest='generate_point', action='store_true')
    # parser.add_argument('--foo', type=int, default=42, help='FOO!')
    parser.set_defaults(generate_point=False)
    args = parser.parse_args()

    generate_map = GenerateMap(start_pickle=1126)

    if args.generate_point:
        generate_map.write_to_pickle()
    else:
        # threading.Thread(target=generate_map.get_local_map).start()
        threading.Thread(target=generate_map.point_generator).start()
        generate_map.read_from_pickle()



