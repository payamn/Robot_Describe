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

from script.utility import Utility
from script.model import Map_Model
from script import constants
from utils import model_utils

class DescribeOnline:

    def __init__(self, resume_path, use_cuda=True):
        self.is_turning = 1
        self.laser_data = None
        self.local_map = None
        self.img_sub = rospy.Subscriber("/cost_map_node/img", Image, self.callback_map_image, queue_size=1)
        self.odom_sub = rospy.Subscriber("/husky_velocity_controller/odom", Odometry, self.callback_odom, queue_size=1)
        self.map = {}
        self.image_pub = rospy.Publisher("local_map", Image, queue_size=1)
        self.bridge = CvBridge()
        self.map_model = Map_Model(resume_path=resume_path)
        self.map_model.model.eval()
        self.use_cuda = use_cuda
        print ("init done")
    def run_network(self, plot=True):

        if self.laser_data is None:
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
    def callback_odom(self, odom):
        # print ("odom:", odom.twist.twist.angular.z)
        if math.fabs(odom.twist.twist.angular.z) > 0.5:
            self.is_turning = 1

    def callback_map_image(self, image):
        # print ("in call back image")
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image, "mono8")
        except CvBridgeError as e:
            print(e)
            return
        if self.is_turning > 0:
            self.is_turning -= 1
            return
        # if self.is_turning > 0:
        #     self.is_turning -= 1
        #     return
        map_array = Utility.normalize(cv_image)
        self.map["data"] = map_array
        # print ("local map updated")
        # print ("get local map")

        self.get_local_map()
        self.run_network()

    def get_local_map(self, annotated_publish=True):
        self.map["info"] = model_utils.get_map_info()
        img = model_utils.get_local_map(
            self.map["info"], self.map["data"], image_publisher=self.image_pub, dilate=True,
            map_dim=constants.LOCAL_MAP_DIM)
        self.local_map = cv2.flip(img, 0)

    def callback_laser_scan(self, scan):
        self.laser_data = [float(x) / scan.range_max for x in scan.ranges]

if __name__=="__main__":
    parser = argparse.ArgumentParser(
        description='online describe')

    parser.add_argument(
        '--resume', metavar='resume', type=str,
        help='resume checkpoint')
    parser.set_defaults(
        resume=os.path.join(rospkg.RosPack().get_path('robot_describe'), 'checkpoints', 'model_best_validation_accuracy_classes.pth.tar'))





    args = parser.parse_args()

    rospy.init_node('Describe_Online')



    describe_online = DescribeOnline(args.resume, torch.cuda.is_available())
    rospy.Subscriber("/scan", LaserScan, describe_online.callback_laser_scan, queue_size=1)
    rospy.spin()