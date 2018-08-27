import cv2 as cv
from cv_bridge import CvBridge
import numpy as np
import matplotlib.pyplot as plt

from script.utility import Utility
from script import constants

import torch
import math

from nav_msgs.srv import GetMap
import rospy

import time

class WordEncoding:
    def __init__(self):
        self.sos = "[sos]"
        self.eos = "[eos]"
        self.sentences = ["close_room", "open_room", "4_junction" ,"t_junction", "corner"]
        classes = ["close_room", "open_room",
                        "corner_left", "corner_right",
                        "t_junction_right_forward", "t_junction_right_left", "t_junction_left_forward", "t_junction", "4_junction"]

        self.classes = {char: idx for idx, char in enumerate(classes)}
        self.classes_labels = {idx: char for idx, char in enumerate(classes)}
        # self.parent_class_dic = {idx: prt_idx for idx, lable in enumerate(classes) for prt_idx, prt_lable in enumerate(self.sentences) if prt_lable in lable}
    def len_classes(self):
        return len(self.classes)

    # def get_parent_class(self, idx):
    #     return self.parent_class_dic[idx]

    def visualize_map(self, map_data, laser_map,predict_classes, predict_poses, predict_objectness, target_classes=None, target_poses=None, target_objectness=None):
        print ("\n\n")
        for batch in range(predict_classes.shape[0]):
            predict = []
            target = []
            # map_data = np.reshape(map_data[batch].cpu().data.numpy(),(map_data.shape[1], map_data.shape[2], 1))
            backtorgb = cv.cvtColor(map_data[batch].cpu().data.numpy(), cv.COLOR_GRAY2RGB)
            backtorgb_laser = cv.cvtColor(laser_map[batch].cpu().data.numpy(), cv.COLOR_GRAY2RGB)
            copy_backtorgb = backtorgb.copy()
            for x in range (predict_classes.shape[1]):
                for y in range (predict_classes.shape[2]):
                    for anchor in range(predict_classes.shape[3]):
                        if target_classes is not None and target_objectness[batch][x][y][anchor].item() >= 0.3:
                            pose = (target_poses[batch][x][y][anchor].cpu().numpy())
                            pose = (pose + np.asarray([x, y])) * ( float(backtorgb.shape[1]) / target_classes.shape[1])
                            pose = pose.astype(int)
                            pose = tuple(pose)
                            target.append((pose, self.get_class_char(target_classes[batch][x][y][anchor].item())))

                            cv.circle(copy_backtorgb, pose, 5, (0, 0, 255))
                            cv.circle(backtorgb_laser, pose, 5, (0, 0, 255))

                        if (predict_objectness[batch][x][y][anchor].item()>= 0.3):
                            pose = ((predict_poses[batch][x][y][anchor].cpu().detach().numpy()))
                            pose = (pose + np.asarray([x, y])) * ( float(backtorgb.shape[1]) / predict_classes.shape[1])
                            pose = pose.astype(int)
                            pose = tuple(pose)
                            predict.append((pose, self.get_class_char(predict_classes[batch][x][y][anchor].item())))
                            cv.circle(backtorgb, pose, 4, (255, 0, 100))
                            cv.circle(backtorgb_laser, pose, 4, (255, 0, 100))
            cv.namedWindow("map")
            cv.namedWindow("laser map")
            cv.imshow("map", copy_backtorgb)
            print ("predict:")
            print predict
            print ("target")
            print (target)
            cv.imshow("laser map", backtorgb_laser)
            cv.waitKey(1    )

            plt.show()

    def get_object_class(self, object):
        if object[1] in self.classes:
            prefer_anchor = 1
            if "room" in object[1]:
                prefer_anchor = 0
            return self.classes[object[1]], object[2], prefer_anchor, object[3]
        else:
            print (object, 0)
            print ("set_objcet_class class not found skiping")

    def get_class_char(self, class_label):
        if class_label in self.classes_labels:
            return self.classes_labels[class_label]
        else:
            return -1


def laser_to_map(laser_array, fov, dest_size, max_range_laser, rotate_degree=None, transform=None, circle_size=3,
                 resize=None):
    fov = float(fov)
    degree_steps = fov/len(laser_array)
    map = np.zeros((dest_size, dest_size, 1))
    to_map_coordinates = float(dest_size)/8.0
    lasers = [((len(laser_array)/2-index)*degree_steps, x) for index, x in enumerate(laser_array)]

    for laser in lasers:
        x = laser[1] * np.cos(laser[0]/180*np.pi) * max_range_laser
        y = laser[1] * np.sin(laser[0]/180*np.pi) * max_range_laser

        x = x * to_map_coordinates
        y = (-y + 8.0/2.) * to_map_coordinates
        if resize:
            # x = x * resize + (1 - resize) * 1 * dest_size
            x = x * resize
            y = y * resize + (1 - resize) * 0.5 * dest_size
        if rotate_degree:
            x, y = Utility.rotate_point((0, dest_size/2.0), (x, y), math.radians(-rotate_degree))
        if transform:
            x, y = (x+transform[0]*dest_size, y+transform[1]*dest_size)
        if x >= dest_size or y >= dest_size or x < 0 or y < 0:
            continue
        cv.circle(map, (int(x), int(y)), circle_size, 255, -1);

        map[int(y), int(x),0] = 255
    # backtorgb = cv.cvtColor(map, cv.COLOR_GRAY2RGB)
    return map


def get_map_info():
    rospy.wait_for_service('/cost_map_node/map_info')
    try:
        map_info_service = rospy.ServiceProxy('/cost_map_node/map_info', GetMap)
        start_time = time.time()
        map_info = map_info_service()
        # print("get map --- %s seconds ---" % (time.time() - start_time))
        return map_info.map.info
        # print "map_info set"
    except rospy.ServiceException, e:
        print ("Service call failed: %s" % e)


def get_map():
    rospy.wait_for_service('/static_map')
    try:
        map_service = rospy.ServiceProxy('/static_map', GetMap)
        start_time = time.time()
        map = map_service()
        # print("get map --- %s seconds ---" % (time.time() - start_time))
        map_array = np.array(map.map.data).reshape((map.map.info.height, map.map.info.width))
        map_array = Utility.normalize(map_array)
        return map_array, map
    except rospy.ServiceException, e:
        print ("Service call failed: %s" % e)

def get_area(x, y, space, local_map):
    start_area = (max(x - space, 0), max(y - space, 0))
    end_area = (min(x + space, local_map.shape[0]), min(y + space, local_map.shape[1]))
    area = local_map[start_area[1]:end_area[1], start_area[0]:end_area[0]]
    mask = np.zeros(area.shape, dtype="uint8")
    cv.circle(mask, (area.shape[0]/2, area.shape[1]/2), area.shape[0]/2, 1, thickness=-1)
    area = mask * area
    return area


def get_local_map(
        map_info, map_data, language=None, image_publisher=None, image_publisher_annotated=None, map_topic_name="map", dilate=False, map_dim=None):

    if map_dim is None:
        map_dim = constants.LOCAL_DISTANCE
    position, quaternion = Utility.get_robot_pose(map_topic_name)
    plus_x = map_info.origin.position.y * map_info.resolution
    plus_y = map_info.origin.position.x * map_info.resolution
    position[0] -= map_info.origin.position.x  # * map_info.resolution
    position[1] -= map_info.origin.position.y  # * map_info.resolution

    _, _, z = Utility.quaternion_to_euler_angle(quaternion[0], quaternion[1],
                                                quaternion[2], quaternion[3])

    image = Utility.sub_image(map_data, map_info.resolution, (position[0], position[1]), z,
                              map_dim, map_dim, only_forward=True, dilate=dilate)

    if image_publisher:
        image_publisher.publish(CvBridge().cv2_to_imgmsg(image))
    local_map = image.copy()

    # if image_publisher_annotated is no`one and language is not None:
    #     for visible in language:
    #         x = int(np.ceil(visible[2][0] / constants.LOCAL_MAP_DIM * image.shape[0]))
    #         y = int(np.ceil((visible[2][1] / (constants.LOCAL_MAP_DIM / 2.0) + 1) / 2 * image.shape[0]))
    #         space = 45
    #         if visible[1] == 'close_room':
    #             space = 10
    #         elif visible[1] == 'open_room':
    #             space = 20
    #         elif 'corner' in visible[1]:
    #             space = 25
    #         start_area = (max(x-space,0), max(y-space,0))
    #         end_area = (min(x+space, image.shape[0]), min(y+space, image.shape[1]))
    #         area_check = local_map[start_area[1]:end_area[1], start_area[0]:end_area[0]]
    #         if np.sum(area_check) > 3*255 or x < image.shape[0]*.30:# pixel
    #             cv.circle(image, (x, y), 4, 100)
    #         else:
    #             print ("empty for ", visible)
    #
    #     image_publisher_annotated.publish(CvBridge().cv2_to_imgmsg(image))
    return local_map
