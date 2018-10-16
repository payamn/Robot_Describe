import rospy
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry

from cv_bridge import CvBridge, CvBridgeError
import torch
from torch.autograd import Variable
import numpy as np
import cv2
import math
import os
import argparse
import rospkg

import pickle
import threading
import time
import subprocess



from script.generate_map import GenerateMap
from script.generate_map import closest_node
from script.utility import Utility
from script.model import Map_Model
from script import constants
from utils import model_utils

MAP_NAME = None
OFFSET_MAP = None
MODE = None
class DescribeOnline:

    def __init__(self, resume_path, use_ground_truth_eval = True, use_cuda=True, mode="full"):
        rospy.init_node('Describe_Online')
        self.init = False
        self.is_turning = 1
        self.laser_data = None
        self.local_map = None
        self.use_ground_truth_eval = use_ground_truth_eval

        try:
            self.odom_topic = rospy.get_param("odom_topic")
        except Exception as e:
            print (e)
            rospy.logerr("odom_topic get error set it to /husky_velocity_controller/odom: ")
            self.odom_topic = "/husky_velocity_controller/odom"

        self.map = {}
        self.image_pub = rospy.Publisher("local_map", Image, queue_size=1)
        self.bridge = CvBridge()
        self.mode = mode
        self.map_model = Map_Model(resume_path=resume_path, mode=self.mode)
        self.map_model.model.eval()
        self.use_cuda = use_cuda
        self.language_gt = {}
        self.finish = False


        self.laser_sub = rospy.Subscriber("/scan", LaserScan, self.callback_laser_scan, queue_size=1)
        self.img_sub = rospy.Subscriber("/cost_map_node/img", Image, self.callback_map_image, queue_size=1)
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.callback_odom, queue_size=1)
        if self.use_ground_truth_eval:
            self.generate_map = GenerateMap(is_online=True, map_offset=OFFSET_MAP, mode=MODE, map_name=MAP_NAME)
            self.t1 = threading.Thread(target=self.generate_map.point_generator)
            self.t1.start()
            self.t2 = threading.Thread(target=self.generate_map.read_from_pickle)
            self.t2.start()
            while not self.generate_map.is_init:
                print ("waiting for generate map to init")
                time.sleep(10)
            print ("init ground truth eval done")
        self.init = True

        print ("init done")


    def run_network(self, plot=True):

        if self.laser_data is None or not self.init:
            return

        center = (0, constants.LOCAL_DISTANCE/ 2)

        laser_scan_map = model_utils.laser_to_map(self.laser_data, constants.LASER_FOV, 240, constants.MAX_RANGE_LASER)
        laser_scan_map = torch.from_numpy(laser_scan_map).type(torch.FloatTensor)

        local_maps = cv2.resize(self.local_map, (240, 240))

        local_maps = torch.from_numpy(local_maps).type(torch.FloatTensor)

        local_maps = Variable(local_maps).cuda() if self.use_cuda else Variable(local_maps)
        laser_scan_map = laser_scan_map.squeeze(2)
        laser_scan_map = Variable(laser_scan_map).cuda() if self.use_cuda else Variable(laser_scan_map)

        local_maps = local_maps.unsqueeze(0)
        local_maps = local_maps.unsqueeze(1)
        laser_scan_map = laser_scan_map.unsqueeze(0)
        laser_scan_map = laser_scan_map.unsqueeze(1)
        # laser_scan_map = laser_scan_map.unsqueeze(0)
        classes_out, poses, objectness = self.map_model.model(laser_scan_map, local_maps)

        topv, topi = classes_out.data.topk(1)

        if plot:
            self.map_model.word_encoding.visualize_map(local_maps[:, 0, :, :], laser_scan_map[:, 0, :, :], (topi, topv), poses, objectness)
    def callback_odom(self, odom_data):
        if not self.init:
            return
        if self.use_ground_truth_eval:
            t = self.generate_map.tf_listner.getLatestCommonTime(self.generate_map.base_frame, "/map_server")
            position, quaternion = self.generate_map.tf_listner.lookupTransform("/map_server", self.generate_map.base_frame, t)

            pose = position
            robot_orientation_degree = Utility.quaternion_to_euler_angle(
                odom_data.pose.pose.orientation.x,
                odom_data.pose.pose.orientation.y,
                odom_data.pose.pose.orientation.z,
                odom_data.pose.pose.orientation.w)

            # new pose is half of the distance we want to cover for language topic
            pose_robot_start_of_image = [
                pose[0] + math.cos(robot_orientation_degree[2] * math.pi / 180) * constants.LOCAL_DISTANCE / 2.0,
                pose[1] + math.sin(robot_orientation_degree[2] * math.pi / 180) * constants.LOCAL_DISTANCE / 2.0,
            ]

            language = closest_node(np.asarray([pose_robot_start_of_image[0], pose_robot_start_of_image[1]]),
                                        self.generate_map.points_description, robot_orientation_degree,
                                        constants.LOCAL_DISTANCE / 2.0
                                        , self.generate_map.tf_listner,
                                        target_frame="map_server_frame")

            return_dic = self.map_model.word_encoding.update_language_ground_truth(language, self.generate_map.finish_all)

            if self.generate_map.finish_all:
                pickle.dump(return_dic, open(os.path.join("map_results/", MAP_NAME + "_" + self.mode + "_" +
                                                          MODE), "wb"))
                self.finish = True
                self.init = False
                self.laser_sub.unregister()
                self.img_sub.unregister()
                self.odom_sub.unregister()
                self.t1._Thread__stop()
                self.t2._Thread__stop()

        self.map_model.word_encoding.near_robot_classes()

        # if self.use_ground_truth_eval:
        #     # save ground truth language:
        #     self.generate_map.callback_robot_0(odom)

    def callback_map_image(self, image):
        if not self.init:
            return
        # print ("in call back image")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image, "mono8")
        except CvBridgeError as e:
            print(e)
            return
        # if self.is_turning > 0:
        #     self.is_turning -= 1
        #     return
        if self.is_turning > 0:
            self.is_turning -= 1
            return
        map_array = Utility.normalize(cv_image)
        self.map["data"] = map_array
        # print ("local map updated")
        # print ("get local map")

        self.get_local_map()
        self.run_network()

    def get_local_map(self, annotated_publish=True):
        if not self.init:
            return
        self.map["info"] = model_utils.get_map_info()
        img = model_utils.get_local_map(
            self.map["info"], self.map["data"], image_publisher=self.image_pub, dilate=True,
            map_dim=constants.LOCAL_MAP_DIM)
        # self.local_map = cv2.flip(img, 0)
        self.local_map = img

    def callback_laser_scan(self, scan):
        self.laser_data = [float(x) / scan.range_max for x in scan.ranges]

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='online describe')

    parser.add_argument(
        '--map_index', metavar='map_index', type=int,
        help='map_index')
    parser.add_argument(
        '--model', metavar='model', type=str,
        help='resume checkpoint')
    parser.set_defaults(model="laser")
    parser.set_defaults(map_index=3)

    args = parser.parse_args()
    f = open("script/data/map.info", "r")
    maps = []
    line = f.readline()
    while (line):
        maps.append(line.split())
        line = f.readline()

    maps = [(x[0], float(x[1]), float(x[2]), x[3]) for x in maps]

    models = {"full":"checkpoints_final_my_resnet/model_best_epoch_accuracy_classes.pth.tar",
                   "laser":"checkpoints_laser_final/model_best_epoch_accuracy_classes.pth.tar",
                   "map":"checkpoints_map_final/model_best_epoch_accuracy_classes.pth.tar"}
    model = args.model
    map = maps[args.map_index]
    MODE = map[3]
    process_str = 'roslaunch robot_describe data_generator.launch transform_pose:="' + str(map[1]) + ' ' + \
                  str(map[2]) + '"  map_name:=' + map[0]
    print process_str
    p = subprocess.Popen(process_str, stdout=None, shell=True)
    time.sleep(10)

    global MAP_NAME, OFFSET_MAP, MODE
    # for model in models:



    MAP_NAME = map[0]
    OFFSET_MAP = (map[1], map[2])

    describe_online = DescribeOnline(models[model], use_cuda=torch.cuda.is_available(), mode=model)


    while True:
        nodes = os.popen("rosnode list").readlines()
        if describe_online.finish or len(nodes) < 4:
            # killall ros nodes
            nodes = os.popen("rosnode list").readlines()
            for i in range(len(nodes)):
                nodes[i] = nodes[i].replace("\n", "")

            for node in nodes:
                if "Describe_Online" not in node:
                    os.system("rosnode kill " + node)
                time.sleep(10)
            # describe_online = DescribeOnline(args.resume, use_cuda=torch.cuda.is_available())
            exit(0)
        time.sleep(5)
        #killall ros nodes
        # nodes = os.popen("rosnode list").readlines()
        # for i in range(len(nodes)):
        #     nodes[i] = nodes[i].replace("\n", "")
        #
        # for node in nodes:
        #     if "Describe_Online" not in node:
        #         os.system("rosnode kill " + node)
        #     time.sleep(10)
    # describe_online = DescribeOnline(args.resume, use_cuda=torch.cuda.is_available())


