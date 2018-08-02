import rospy
from sensor_msgs.msg import LaserScan, Image
from nav_msgs.msg import Odometry

from cv_bridge import CvBridge, CvBridgeError
import torch
from torch.autograd import Variable
import numpy as np
import cv2
import math

from script.utility import Utility
from script.model import Map_Model
from script import constants
from utils import model_utils

class DescribeOnline:

    def __init__(self, use_cuda=True):
        self.is_turning = 1
        self.laser_data = None
        self.local_map = None
        self.img_sub = rospy.Subscriber("/cost_map_node/img", Image, self.callback_map_image, queue_size=1)
        self.odom_sub = rospy.Subscriber("/odom", Image, self.callback_odom, queue_size=1)
        self.map = {}
        self.image_pub = rospy.Publisher("local_map", Image, queue_size=1)
        self.bridge = CvBridge()
        self.map_model = Map_Model()
        self.use_cuda = use_cuda
    def run_network(self, plot=True):

        if self.laser_data is None:
            return

        laser_scan_map = model_utils.laser_to_map(self.laser_data, constants.LASER_FOV, 240, constants.MAX_RANGE_LASER)
        laser_map= torch.from_numpy(np.stack([laser_scan_map, laser_scan_map, laser_scan_map])).type(torch.FloatTensor)

        local_maps = cv2.resize(self.local_map, (240, 240))

        laser_scan_map = laser_scan_map.squeeze(2)
        local_map = torch.from_numpy(np.stack([local_maps, laser_scan_map, laser_scan_map])).type(torch.FloatTensor)

        local_map = Variable(local_map).cuda() if self.use_cuda else Variable(local_map)
        laser_map = Variable(laser_map).cuda() if self.use_cuda else Variable(laser_map)

        local_map = local_map.unsqueeze(0)
        laser_map = laser_map.unsqueeze(0)
        classes_out, poses, objectness = self.map_model.model(local_map)

        topv, topi = classes_out.data.topk(1)

        if plot:
            self.map_model.word_encoding.visualize_map(local_map[:, 0, :, :], laser_map[:, 0, :, :], topi, poses, objectness)
    def callback_odom(self, odom):
        if math.fabs(odom.twist.angular.z) > 0.2:
            self.is_turning = 4

    def callback_map_image(self, image):
        print ("in call back image")
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
        print ("get local map")

        self.get_local_map()
        self.run_network()

    def get_local_map(self, annotated_publish=True):
        self.map["info"] = model_utils.get_map_info()
        self.local_map = model_utils.get_local_map(
                self.map["info"], self.map["data"], image_publisher=self.image_pub)


    def callback_laser_scan(self, scan):
        self.laser_data = [float(x) / scan.range_max for x in scan.ranges]

if __name__=="__main__":
    rospy.init_node('Describe_Online')

    describe_online = DescribeOnline()
    rospy.Subscriber("/base_scan", LaserScan, describe_online.callback_laser_scan, queue_size=1)
    rospy.spin()