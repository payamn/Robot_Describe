import cv2 as cv
from cv_bridge import CvBridge
import numpy as np
from geometry_msgs.msg import Pose, PoseArray
import tf
import matplotlib.pyplot as plt
from tf import TransformerROS
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
        # self.sentences = ["close_room", "open_room", "corridor"]
        classes = ["close_room", "open_room", "corridor"]

        self.classes = {char: idx for idx, char in enumerate(classes)}
        self.classes_labels = {idx: char for idx, char in enumerate(classes)}
        # self.parent_class_dic = {idx: prt_idx for idx, lable in enumerate(classes) for prt_idx, prt_lable in enumerate(self.sentences) if prt_lable in lable}
        #TODO: Move next line
        self.publisher = {
            "close_room" : rospy.Publisher('close_rooms', PoseArray, queue_size=10, latch=True),
            "open_room": rospy.Publisher('open_rooms', PoseArray, queue_size=10, latch=True),
            "corridor": rospy.Publisher('corridor', PoseArray, queue_size=10, latch=True)
        }
        self.publish_list = {
            "close_room": [],
            "open_room": [],
            "corridor": []
        }
        self.tf_ros = TransformerROS()

    def len_classes(self):
        return len(self.classes)

    # def get_parent_class(self, idx):
    #     return self.parent_class_dic[idx]

    def closest_node(self, node, nodes, limit):
        return_nodes = []
        if len(nodes) == 0:
            return return_nodes
        nodes_array = np.asarray(nodes)[:, 0:2]
        dist = np.sqrt(np.sum((nodes_array - node[0:2]) ** 2, axis=1))
        for index, node_dist in enumerate(dist):
            if node_dist < limit:
                return_nodes.append((index, node_dist))
        return return_nodes

    def appropriate_point_class(self, point, objectness,class_lable):
        not_publish_point = 0
        for mode in self.classes:
            if mode not in self.publish_list:
                continue

            distance = 0.7
            if class_lable == 2:
                distance = 1.2

            closest_points = self.closest_node(point, self.publish_list[mode], distance)
            if len(closest_points) == 0:
                continue
            del_nodes = []
            for close_point in closest_points:
                if class_lable == self.classes[mode]:
                    point = ((point[0] + self.publish_list[mode][close_point[0]][0])/2,
                             (point[1] + self.publish_list[mode][close_point[0]][1])/2)
                    objectness = (self.publish_list[mode][close_point[0]][2] + objectness)/2
                    del_nodes.append(close_point[0])
                    not_publish_point += 1
                elif class_lable != self.classes[mode] and close_point[1] < 0.3:
                    if objectness < self.publish_list[mode][close_point[0]][2]:

                        self.publish_list[mode][close_point[0]] = (
                            self.publish_list[mode][close_point[0]][0],
                            self.publish_list[mode][close_point[0]][1],
                            self.publish_list[mode][close_point[0]][2] * 0.95,
                            self.publish_list[mode][close_point[0]][3]
                        )

                        not_publish_point -= 1

                    else:
                        if self.publish_list[mode][close_point[0]][2] < 0.2:
                            del_nodes.append(close_point[0])
                        else:
                            pass
                        self.publish_list[mode][close_point[0]] = (self.publish_list[mode][close_point[0]][0],
                                                                   self.publish_list[mode][close_point[0]][1],
                                                                   self.publish_list[mode][close_point[0]][2] * 0.85,
                                                                   self.publish_list[mode][close_point[0]][3]
                                                                   )

                        objectness = objectness * 0.85

            for node in del_nodes:
                self.publish_list[mode].__delitem__(node)

        if not_publish_point < 0:
            return None
        return (point[0], point[1], objectness, class_lable)

    def publish_point_around_robot(self, points):
        """

        :param points: dictionary of class and points
        """
        position_robot, quaternion_robot = Utility.get_robot_pose("map")
        mat44 = self.tf_ros.fromTranslationRotation(position_robot, quaternion_robot)

        for mode in self.classes:
            if mode in points:
                for point in points[mode]:
                    point_xy = tuple(np.dot(mat44, np.array([point[0], point[1], 0, 1.0])))[:2]
                    point = self.appropriate_point_class(point_xy, point[2], self.classes[mode])
                    if point is not None:
                        self.publish_list[mode].append(point)
        pose_msgs = {}
        for mode in self.classes:
            if mode not in self.publish_list:
                continue
            pose_msgs[mode] = PoseArray()
            pose_msgs[mode].header.frame_id = 'map'
            pose_msgs[mode].header.stamp = rospy.Time.now()
            for point in self.publish_list[mode]:
                if point[2] < 0.3:
                    continue
                pose_msg = Pose()
                pose_msg.position.x = point[0]
                pose_msg.position.y = point[1]
                pose_msg.position.z = 0
                w, x, y, z = Utility.toQuaternion(
                   0, 0, 0)
                pose_msg.orientation.w = w
                pose_msg.orientation.x = x
                pose_msg.orientation.y = y
                pose_msg.orientation.z = z
                pose_msgs[mode].poses.append(pose_msg)

            self.publisher[mode].publish(pose_msgs[mode])

    def visualize_map(self, map_data, laser_map,predict_classes, predict_poses, predict_objectness, target_classes=None, target_poses=None, target_objectness=None):
        print ("\n\n")
        for batch in range(predict_classes[0].shape[0]):
            predict = []
            target = []
            # map_data = np.reshape(map_data[batch].cpu().data.numpy(),(map_data.shape[1], map_data.shape[2], 1))
            backtorgb = cv.cvtColor(map_data[batch].cpu().data.numpy(), cv.COLOR_GRAY2RGB)
            backtorgb_laser = cv.cvtColor(laser_map[batch].cpu().data.numpy(), cv.COLOR_GRAY2RGB)
            copy_backtorgb = backtorgb.copy()
            publish_poses = {"close_room":[], "open_room":[], "corridor":[]}
            for x in range (predict_classes[0].shape[1]):
                for y in range (predict_classes[0].shape[2]):
                    for anchor in range(predict_classes[0].shape[3]):
                        if target_classes is not None and target_objectness[batch][x][y][anchor].item() >= constants.ACCURACY_THRESHOLD:
                            pose = (target_poses[batch][x][y][anchor].cpu().numpy())
                            pose = (pose + np.asarray([x, y])) * ( float(backtorgb.shape[1]) / target_classes.shape[1])
                            pose = pose.astype(int)
                            pose = tuple(pose)
                            target.append((pose, self.get_class_char(target_classes[batch][x][y][anchor].item())))

                            cv.circle(copy_backtorgb, pose, 5, (0, 0, 255))
                            cv.circle(backtorgb_laser, pose, 5, (0, 0, 255))

                        if (predict_objectness[batch][x][y][anchor].item()>= constants.ACCURACY_THRESHOLD):
                            pose = ((predict_poses[batch][x][y][anchor].cpu().detach().numpy()))
                            pose_map = (pose + np.asarray([x, y])) * ( float(constants.LOCAL_MAP_DIM) / predict_classes[0].shape[1]) -np.asarray([0, constants.LOCAL_MAP_DIM/2.0])
                            pose_map = (pose_map[0], pose_map[1], predict_objectness[batch][x][y][anchor].item() * math.exp(predict_classes[1][batch][x][y][anchor].item()))
                            # pose_map[1] = -pose_map[1]
                            pose = (pose + np.asarray([x, y])) * ( float(backtorgb.shape[1]) / predict_classes[0].shape[1])
                            pose = pose.astype(int)
                            pose = tuple(pose)
                            predict.append((pose, self.get_class_char(predict_classes[0][batch][x][y][anchor].item()),
                                            predict_objectness[batch][x][y][anchor].item() * math.exp(predict_classes[1][batch][x][y][anchor].item())))
                            color = (255, 0, 100)
                            if predict_classes[0][batch][x][y][anchor].item() == 1:
                                publish_poses["open_room"].append(pose_map)
                                color = (100,0,255)
                            elif predict_classes[0][batch][x][y][anchor].item()  == 0:
                                publish_poses["close_room"].append(pose_map)
                                color = (255,0,0)
                            elif predict_classes[0][batch][x][y][anchor].item()  == 2:
                                publish_poses["corridor"].append(pose_map)
                                color = (255,0,0)
                            cv.circle(backtorgb, pose, 4, color)
                            cv.circle(backtorgb_laser, pose, 4, color)
            self.publish_point_around_robot(publish_poses)
            cv.namedWindow("map")
            cv.namedWindow("laser map")
            cv.imshow("map", backtorgb)
            print ("predict:")
            print predict
            print ("target")
            print (target)
            cv.imshow("laser map", backtorgb_laser)
            cv.waitKey(1        )

            plt.show()

    def get_object_class(self, object):
        # no anchor anymore
        if object[1] in self.classes:
            return self.classes[object[1]], object[2], 0, object[3]
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

