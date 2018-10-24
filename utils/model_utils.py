import cv2 as cv
from cv_bridge import CvBridge
import numpy as np
from geometry_msgs.msg import Pose, PoseArray
from visualization_msgs.msg import Marker
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
        self.marker_publisher = rospy.Publisher('mark', Marker, queue_size=1, latch=True)
        self.id_marker = 1
        self.classes_acc_track = {"close_room": [], "open_room": [], "corridor": []}
        self.no_gt_matched = {"close_room": [], "open_room": [], "corridor": []}
        self.classes_distance = {"close_room": [], "open_room": [], "corridor": []}
        self.points = []
        self.tf_ros = TransformerROS()
        self.language_gt = {}
        self.frame_origin = "/map_server"
    def len_classes(self):
        return len(self.classes)

    def re_init(self):
        self.classes_acc_track = {"close_room": [], "open_room": [], "corridor": []}
        self.no_gt_matched = {"close_room": [], "open_room": [], "corridor": []}
        self.classes_distance = {"close_room": [], "open_room": [], "corridor": []}
        self.points = []
        self.language_gt = {}

    def del_old_nodes(self):
        now = time.time()
        del_nodes = []
        for x in range(len(self.points)):
            # after 30 second remove point
            if now - self.points[x][0][3] > constants.POINT_LIFE_TIME:
                del_nodes.append(x)
                # print ("del node in closest node")
                continue
        if len(del_nodes) > 0:
            self.points = [x for idx, x in enumerate(self.points) if idx not in del_nodes]

    def update_language_ground_truth(self, langs, is_last = False):
        """

        :param langs: all the language topics detected so far to validate them using ground truth
        :param is_last: if it is the last call of this function to calculate ground truth for all the language topics
        :return: return is a dictionary containing accuracy untill this point
        """

        return_dic = {}
        position_robot, quaternion_robot = Utility.get_robot_pose(self.frame_origin)
        for lang in langs:
            if not lang[0] in self.language_gt:
                # lang[0] is hash key, 0 (lang[1]): class, 1 (lang[2]): position, 2:matched class 3: distance matched 4: if distance to point is less than 4 meters(to have to be detected)
                self.language_gt[lang[0]] = [lang[1], lang[2], None, None, False]
        lang_del = []
        for lang in self.language_gt:
            distance = Utility.distance_vector(self.language_gt[lang][1], position_robot)

            # if self.language_gt[lang][0] == "open_room":
            #     print (lang, distance, self.language_gt[lang])

            if distance < 4:
                self.language_gt[lang][4] = True

            # check to remove the point
            if (distance > 8 or is_last) and self.language_gt[lang][4]:
                # if self.language_gt[lang][0] == "open_room":
                    # print ("open_room")
                self.classes_acc_track[self.language_gt[lang][0]] .append(self.language_gt[lang][2])
                lang_del.append(lang)
                if self.language_gt[lang][2] is not None and self.language_gt[lang][0] == self.classes_labels[self.language_gt[lang][2]]:
                    self.classes_distance[self.language_gt[lang][0]].append(self.language_gt[lang][3])
                    # print ("mean distance:", self.language_gt[lang][0], np.mean(self.classes_distance[self.language_gt[lang][0]]))
            elif distance < 4:
                self.language_gt[lang][4] = True
        # print  self.classes_acc_track
        mean_acc = {lang:[] for lang in  self.classes_acc_track}
        for lang in  self.classes_acc_track:
            no_gt_matched = len(self.no_gt_matched[lang])
            each_acc_classes = [0, 0, 0, 0]
            for node in self.classes_acc_track[lang]:
                if node is None:
                    each_acc_classes[3] += 1
                else:
                    each_acc_classes[node] += 1
            if len(self.classes_acc_track[lang]) ==0:
                if is_last:
                    print (lang, "len 0 no matched:", len(self.no_gt_matched[lang]))
                continue
            if len(self.no_gt_matched[lang]) > 0:
                change = min(len(self.no_gt_matched[lang]), each_acc_classes[3], 1)
                each_acc_classes[self.classes[lang]] += change
                each_acc_classes[3] -= change
                no_gt_matched = len(self.no_gt_matched[lang]) - change

            each_acc_classes_percent = [x/float(len(self.classes_acc_track[lang])) for x in each_acc_classes ]

            return_dic[lang] = {"each_acc_classes_percent":each_acc_classes_percent,
                                "classes_acc_track":self.classes_acc_track[lang],
                                "each_acc_classes": each_acc_classes,
                                "distance":self.classes_distance[lang],
                                "no_matched":no_gt_matched
                                }
            if len(self.classes_distance[lang]) == 0:
                self.classes_distance[lang].append(-1)
            if is_last:
                print (lang, each_acc_classes_percent, "distance min, max, avg:", np.min(self.classes_distance[lang]),
                       np.max(self.classes_distance[lang]), np.mean(self.classes_distance[lang]), each_acc_classes, len(self.classes_acc_track[lang]))
                print ("no matched", no_gt_matched)
                print ("")

        for lang in lang_del:
            self.language_gt.__delitem__(lang)

        return return_dic
    def closest_node(self, node, limit):



        return_nodes = []


        for x in range(len(self.points)):
            node_dist = math.sqrt((node[0] - self.points[x][0][0][0]) ** 2 +   (node[1] - self.points[x][0][0][1]) ** 2)
            if node_dist < limit:
                return_nodes.append(x)

        # if len(nodes) == 0:
        #     return return_nodes
        # nodes_array = np.asarray(nodes)[:, 0:2]
        # dist = np.sqrt(np.sum((nodes_array - node[0:2]) ** 2, axis=1))
        # for index, node_dist in enumerate(dist):
        #     if node_dist < limit:
        #         return_nodes.append((index, node_dist))
        return return_nodes

    def calculate_avg_one_group(self, idx):
        each_class = [[] for x in range(3)]  # point, objectness fo each 3 class
        for point in self.points[idx][1:]:
            each_class[point[2]].append(point)
        each_class_point = [None for x in range(3)]
        max_objectness_class = None
        for i in range(3):
            if len(each_class[i]) == 0:
                continue
            avg_point = (0, 0)
            avg_objectness = 0
            for point in each_class[i]:
                avg_objectness += point[1]
                avg_point = (avg_point[0] + point[0][0], avg_point[1] + point[0][1])
            avg_objectness = avg_objectness / len(each_class[i])
            avg_point = (avg_point[0] / len(each_class[i]), avg_point[1] / len(each_class[i]))
            each_class_point[i] = (avg_point, avg_objectness)
            if max_objectness_class is None or max_objectness_class[1] < avg_objectness:
                max_objectness_class = (i, avg_objectness)

        matched_gt = self.matched_ground_truth_class(each_class_point[max_objectness_class[0]][0], max_objectness_class[0],
                                                     prev_match_gt = self.points[idx][0][5], prev_class_lable = self.points[idx][0][3])


        seen_point = False
        # if already printed the point around robot and class not changed
        if self.points[idx][0][4] and self.points[idx][0][3] != max_objectness_class[0]:
            seen_point = True
        self.points[idx][0] = [each_class_point[max_objectness_class[0]][0], max_objectness_class[1],
                               max_objectness_class[0], time.time(), seen_point, matched_gt]
    def append_to_points(self, idx, point, objectness, class_lable, distance):
        self.points[idx].append([point, objectness, class_lable])
        self.calculate_avg_one_group(idx)
        closest_groups = self.closest_node(self.points[idx][0][0], distance)

        if len(closest_groups) == 0:
            return
        del_nodes = []
        for group in closest_groups:
            if group == idx:
                continue
            if self.points[group][0][1] < 0.15:
                del_nodes.append(group)
                # print ("dell group")
            elif self.points[idx][0][1] < 0.15:
                del_nodes.append(idx)
                # print ("dell group")

            else:
                for node in self.points[group][1:]:
                    self.points[idx].append(node)
                    # print ("dell group")

                del_nodes.append(group)


        self.points = [x for idx, x in enumerate(self.points) if idx not in del_nodes]

    def matched_ground_truth_class(self, point, class_lable, prev_match_gt=None, prev_class_lable=None):



        position_robot, quaternion_robot = Utility.get_robot_pose(self.frame_origin)

        matched = "Nothing"
        desire_dist_same_class = 2
        desire_dist_different_class = 1.2

        min_distance = 10000
        for lang in self.language_gt:
            distance =  Utility.distance_vector(self.language_gt[lang][1], point)
            if (self.language_gt[lang][2] == class_lable and distance < desire_dist_same_class) or (self.language_gt[lang][2] != class_lable and distance < desire_dist_different_class):
                change = False
                if matched == "Nothing":
                    change = True
                elif self.language_gt[matched][2] != class_lable and (min_distance > distance or self.language_gt[lang][2] == class_lable):
                    change = True
                elif self.language_gt[matched][2] == class_lable and self.language_gt[lang][2] == class_lable and min_distance > distance:
                    change = True
                if change:
                    matched = lang
                    min_distance = distance
        if prev_match_gt == matched:
            return matched

        distance_to_robot = None
        if matched != "Nothing":
            distance_to_robot = Utility.distance_vector(self.language_gt[matched][1], position_robot)



        if matched == "Nothing" and prev_match_gt is None:
            self.no_gt_matched[self.classes_labels[class_lable]].append(1)

        elif matched == "Nothing" and prev_match_gt is not None:
            self.language_gt[prev_match_gt][2] = None
            self.language_gt[prev_match_gt][3] = None

        elif prev_match_gt == "nothing" and prev_class_lable is not None:
            #if it happens means prev was nothing and this one is a class
            self.no_gt_matched[self.classes_labels[prev_class_lable]] = self.no_gt_matched[self.classes_labels[prev_class_lable]][0:-1]
            self.language_gt[matched][2] = class_lable
            self.language_gt[matched][3] = distance_to_robot
        elif prev_match_gt == "nothing" and prev_class_lable is None:
            self.language_gt[matched][2] = class_lable
            self.language_gt[matched][3] = distance_to_robot
            print ("error  prev_match_gt == nothing and prev_class_lable is None")
        elif prev_match_gt is None:
            # Normal case after first match
            self.language_gt[matched][2] = class_lable
            self.language_gt[matched][3] = distance_to_robot

        else:
            self.language_gt[matched][2] = class_lable
            self.language_gt[matched][3] = distance_to_robot

        return matched

    def appropriate_point_class(self, point, objectness,class_lable):
        not_publish_point = 0

        distance = 1.3
        self.del_old_nodes()
        closest_points = self.closest_node(point, distance)

        if len(closest_points) == 0:
            matched_gt = self.matched_ground_truth_class(point, class_lable)
            self.points.append([[point, objectness, class_lable, time.time(), False, matched_gt], [point, objectness, class_lable]])
        elif len(closest_points) == 1:
            self.append_to_points(closest_points[0], point, objectness, class_lable, distance)
        else:
            self.append_to_points(closest_points[0], point, objectness, class_lable, distance)

    def near_robot_classes(self):
        position_robot, quaternion_robot = Utility.get_robot_pose(self.frame_origin)

        euler_r = tf.transformations.euler_from_quaternion(quaternion_robot)
        # perpendicular_r = euler_r[2] + math.pi / 2
        # perpendicular_r_q = tf.transformations.quaternion_from_euler(0, 0, perpendicular_r)
        # mat44 = self.tf_ros.fromTranslationRotation(position_robot, quaternion_robot)
        # mat44_p = self.tf_ros.fromTranslationRotation(position_robot, perpendicular_r_q)
        # range = (3.2,4)

        # x1, x2 = tuple(np.dot(mat44, np.array((range[0] , 0, 0, 1))))[:2], tuple(np.dot(mat44, np.array((-0.5, 0, 0, 1))))[:2]
        # range_x_w_temp = (tuple(np.dot(mat44, np.array((range[0]*2 , 0, 0, 1))))[:2], tuple(np.dot(mat44, np.array((-0.5, 0, 0, 1))))[:2])
        # y1, y2 = tuple(np.dot(mat44, np.array((0, range[1], 0, 1))))[:2], tuple(np.dot(mat44, np.array((0,-range[1], 0, 1))))[:2]
        #
        # range_x = (max(range_x_temp[0][0], range_x_temp[1][0], range_y_temp[0][0], range_y_temp[1][0]),
        #            min(range_x_temp[0][0], range_x_temp[1][0], range_y_temp[0][0], range_y_temp[1][0]) )
        # # range_x_w = (max(range_x_w_temp[0][0], range_x_w_temp[1][0]), min(range_x_w_temp[0][0], range_x_w_temp[1][0]))
        # range_y = (max(range_y_temp[0][1], range_y_temp[1][1], range_x_temp[0][1], range_x_temp[1][1]),
        #            min(range_y_temp[0][1], range_y_temp[1][1], range_x_temp[0][1], range_x_temp[1][1]))

        for index, point in enumerate (self.points):
            distance =  math.sqrt(math.pow(point[0][0][0] - position_robot[0], 2) +  math.pow(point[0][0][1] - position_robot[1], 2))
            if not point[0][4] and point[0][1] > 0.15:
                angle = (euler_r[2] - math.atan2(point[0][0][1] - position_robot[1],
                                                 point[0][0][0] - position_robot[0])) * 180 / math.pi

                if angle > 180:
                    angle -= 360
                elif angle < -180:
                    angle += 360

                position_str = None
                if distance < 3.5 :
                    # or\
                        # (not point[0][4] and point[0][2] == 2 and point[0][0][0] > range_x_w[1] and point[0][0][0] < range_x_w[0]\
                        # and point[0][0][1] > range_y[1] and point[0][0][1] < range_y[0] ):


                    if  angle > -120 and angle < -50: # on the left side
                        position_str = "left"
                    elif angle > -30 and angle < 30 and (distance < 1 or (point[0][2] == 2 and distance < 2)):
                        position_str = "forward"
                    elif angle > 50 and angle < 120:
                        position_str = "right"

                elif distance < 5 and point[0][2] == 2 :
                    if  angle > -55 and angle < -15: # on the left side
                        position_str = "left"
                    elif angle > -5 and angle < 5 :
                        position_str = "forward"
                    elif angle > 15 and angle < 55:
                        position_str = "right"

                if position_str is not None:
                    print (self.classes_labels[point[0][2]], position_str, angle)
                    marker = Marker()
                    position_robot, quaternion_robot = Utility.get_robot_pose(self.frame_origin)
                    marker.header.frame_id = self.frame_origin
                    marker.id = self.id_marker
                    marker.type = marker.TEXT_VIEW_FACING
                    marker.text = " ".join(self.classes_labels[point[0][2]].split("_")) + " " + position_str
                    marker.action = marker.ADD
                    marker.scale.x = 0.3
                    marker.scale.y = 0.3
                    marker.scale.z = 0.4
                    marker.color.a = 1.0
                    marker.color.r = 0.4
                    marker.color.g = 0.2
                    marker.color.b = 0.1
                    marker.lifetime = rospy.Duration(20)
                    marker.pose.orientation.w = 1.0
                    marker.pose.position.x = (point[0][0][0] + 2 * position_robot[0]) / 3
                    marker.pose.position.y = (point[0][0][1] + 2 * position_robot[1]) / 3
                    marker.pose.position.z = 0
                    self.id_marker += 1
                    self.marker_publisher.publish(marker)
                    self.points[index][0] = (point[0][0], point[0][1], point[0][2], point[0][3], True, point[0][5])
    def publish_point_around_robot(self, points, map_frame="map_server"):
        """

        :param points: dictionary of class and points
        """
        position_robot, quaternion_robot = Utility.get_robot_pose(map_frame)
        mat44 = self.tf_ros.fromTranslationRotation(position_robot, quaternion_robot)
        for mode in self.classes:
            if mode in points:
                for point in points[mode]:
                    point_xy = tuple(np.dot(mat44, np.array([point[0], point[1], 0, 1.0])))[:2]
                    self.appropriate_point_class(point_xy, point[2], self.classes[mode])

        pose_msgs = {}
        publish_list = {
            "close_room": [],
            "open_room": [],
            "corridor": []
        }
        for point in self.points:
            publish_list[self.classes_labels[point[0][2]]].append((point[0][0][0], point[0][0][1], point[0][1]))
        for mode in self.classes:
            if mode not in publish_list:
                continue
            pose_msgs[mode] = PoseArray()
            pose_msgs[mode].header.frame_id = map_frame
            pose_msgs[mode].header.stamp = rospy.Time.now()
            for point in publish_list[mode]:
                if point[2] < 0.45:
                    # objectness less than 0.2
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
        # print ("\n\n")
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

                        if (predict_objectness[batch][x][y][anchor].item()>= constants.ACCURACY_THRESHOLD_PREDICTION):
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
            cv.imshow("laser map", backtorgb_laser)
            # print ("predict:")
            # print predict
            # print ("target")
            # print (target)
            # cv.imshow("laser map", backtorgb_laser)
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
        cv.circle(map, (int(x), int(y)), circle_size, 255, -1)

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

