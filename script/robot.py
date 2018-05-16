import rospkg

import actionlib
import rospy
from geometry_msgs.msg import PoseStamped, Twist
from mbf_msgs.msg import MoveBaseAction
from numpy import *
from script.prev_dataset.dataset_prev import *
from std_msgs.msg import String
from utility import *


class IntersectionNode:
    def __init__(self, pose, connected_nodes):
        self.connection = connected_nodes
        self.pose = pose


class Robot:
    def __init__(self, is_data_generation, data_set_ver, start_index_dataset = 0, batch_size = 20, real_time_test=False, use_direction_file=True, bag_name="/bag_2_test/"):
        random.seed()
        self.speed = (0, 0, 0)
        rospack = rospkg.RosPack()
        self.data_set_ver = data_set_ver
        self.remaining_intersection = 1
        self.path = rospack.get_path('robot_describe')
        self.language_topic = "language"
        self.lock = threading.Lock()
        self.is_stoped = False
        self.not_change = 0
        self.real_time = real_time_test
        # Not yet implemented for dataset version 1
        if real_time_test == True and data_set_ver == 2:
            print "real time test"
        elif data_set_ver == 1:
            self.dataset = DataSet(self.path + "/data/directions.txt", self.path + bag_name, is_data_generation, batch_size = batch_size, bag_number = start_index_dataset, use_direction_file=use_direction_file)
        else:
            self.dataset = DataSet(self.path + "/data/directions.txt", self.path + bag_name, is_data_generation, batch_size = batch_size, bag_number = start_index_dataset, use_direction_file=use_direction_file)

        self.lang_bag_str = String()
        if (use_direction_file):
            self.list_intersection = []
            self.list_intersection.append(IntersectionNode(array([-4+mapper_constant_val_x, mapper_constant_val_y - 4]),
                                                           array([1, 5])))
            self.list_intersection.append(IntersectionNode(array([-4+mapper_constant_val_x, mapper_constant_val_y + 0]),
                                                           array([0, 2, 4])))
            self.list_intersection.append(IntersectionNode(array([-4+mapper_constant_val_x, mapper_constant_val_y + 4]),
                                                           array([1, 3])))
            self.list_intersection.append(IntersectionNode(array([4+mapper_constant_val_x, mapper_constant_val_y + 4]),
                                                           array([2, 4])))
            self.list_intersection.append(IntersectionNode(array([4+mapper_constant_val_x, mapper_constant_val_y + 0]),
                                                           array([1, 3, 5])))
            self.list_intersection.append(IntersectionNode(array([4+mapper_constant_val_x, mapper_constant_val_y - 4]),
                                                           array([0, 4])))
            self.prev_pose = 1  # intersection 1
            self.next_pose = 0  # intersection 0
            self.pose = self.list_intersection[0].pose
        self.publisher = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=10, latch=True)
        self.lock = threading.Lock()
        if is_data_generation:
            self.client = actionlib.SimpleActionClient("move_base_flex/move_base", MoveBaseAction)
            self.remaining_laser_scan = LASER_SCAN_WINDOW  # start with LASER_SCAN_WINDOW which means wait for LASER_SCAN_WINDOW laser scan as the array empty
            self.laser_scans = [LaserScan() for x in range(LASER_SCAN_WINDOW)]  # allocate LASER_SCAN_WINDOW space for LASER_SCAN_WINDOW laser scan
            self.speeds = [Twist() for x in range(LASER_SCAN_WINDOW)]  # allocate LASER_SCAN_WINDOW space for LASER_SCAN_WINDOW laser scan
            self.laser_pointer = 0  # index to add new laser scan
            if (use_direction_file):
                self.send_pos(self.list_intersection[0].pose)
    def set_speed(self, twist_speed):
        self.speed = twist_speed
    def set_stop(self, is_stop):
        self.is_stoped = is_stop
    def send_pos(self, pose):

        target_goal_simple = PoseStamped()

        target_goal_simple.pose.position.x =pose[0]
        target_goal_simple.pose.position.y = pose[1]
        target_goal_simple.pose.position.z = 0
        target_goal_simple.pose.orientation.w = 1
        target_goal_simple.header.frame_id = 'map'
        target_goal_simple.header.stamp = rospy.Time.now()
        self.publisher.publish(target_goal_simple)
        # server_connected = self.client.wait_for_server()
        # while not server_connected:
        #     print ("server not connected try again")
        #     server_connected = self.client.wait_for_server()
        # self.client.cancel_goal()
        # print ("after cancel", self.client.wait_for_result())
        #
        # goal = MoveBaseGoal()
        # goal.target_pose.pose.position.x = pose[0]
        # goal.target_pose.pose.position.y = pose[1]
        # goal.target_pose.pose.position.z = 0
        # goal.target_pose.pose.orientation.w = 1
        # goal.target_pose.header.frame_id = 'map'
        # goal.target_pose.header.stamp = rospy.Time.now()
        #
        # # start moving
        # self.client.send_goal(goal)
    def bag_scan_laser_online(self, scan):
        self.lock.acquire()
        # success = self.client.get_goal_status_text()
        # print('success status', success)
        self.laser_scans[self.laser_pointer] = scan
        self.laser_pointer = (self.laser_pointer + 1) % LASER_SCAN_WINDOW

        # self.remaining_laser_scan -= 1
        # if (self.remaining_laser_scan <= 0):        # getting to limit so save the bag files
        #     if (self.lang_bag_str.data == "nothing" and random.uniform(0, 1.0) > 0.9):
        #         self.remaining_laser_scan = LASER_SCAN_WINDOW / 2
        #         self.lock.release()
        #         return
        #     rospy.logerr(self.lang_bag_str.data)
        #     self.data_set.write_bag(self.language_topic, self.lang_bag_str)
        #     self.remaining_laser_scan = LASER_SCAN_WINDOW / 2
        #     for i in range (LASER_SCAN_WINDOW):
        #         if (not self.data_set.write_bag("/robot_0/base_scan_1", self.laser_scans[(i + self.laser_pointer) % LASER_SCAN_WINDOW])):
        #             self.not_change = 0
        #             self.lock.release()
        #             return
        #
        #     threading.Timer(0.0, self.data_set.new_sequence).start()
        #     if self.not_change > 0:
        #         self.remaining_laser_scan = 2
        #         self.not_change -= 1
        self.lock.release()

    def save_bag_scan_laser(self, scan):
        self.lock.acquire()
        # success = self.client.get_goal_status_text()
        # print('success status', success)
        if self.is_stoped:
            self.lock.release()
            return
        self.laser_scans[self.laser_pointer] = scan
        self.speeds[self.laser_pointer] = self.speed
        self.laser_pointer = (self.laser_pointer + 1) % LASER_SCAN_WINDOW

        self.remaining_laser_scan -= 1
        if (self.remaining_laser_scan <= 0):        # getting to limit so save the bag files
            if (self.lang_bag_str.data == "nothing" and random.uniform(0, 1.0) > 0.9):
                self.remaining_laser_scan = LASER_SCAN_WINDOW / 2
                self.lock.release()
                return

            rospy.logerr(self.lang_bag_str.data)
            self.dataset.write_bag(self.language_topic, self.lang_bag_str)
            self.remaining_laser_scan = LASER_SCAN_WINDOW / 2
            for i in range (LASER_SCAN_WINDOW):
                if (not self.dataset.write_bag("/robot_0/base_scan_1", self.laser_scans[(i + self.laser_pointer) % LASER_SCAN_WINDOW])):
                    self.not_change = 0
                    self.lock.release()
                    return

                self.dataset.write_bag("/robot_0/speed",
                                       self.speeds[(i + self.laser_pointer) % LASER_SCAN_WINDOW])

            threading.Timer(0.0, self.dataset.new_sequence).start()
            if self.not_change > 0:
                self.remaining_laser_scan = 2
                self.not_change -= 1
            else:
                self.lang_bag_str.data = "nothing"

        self.lock.release()

    def reset_intersection_collection(self):
        # self.remaining_intersection = random.randint(3, 7)
        self.remaining_intersection = 1
        threading.Timer(0.5, self.dataset.new_sequence).start()
        # threading.Timer(random.uniform(0.5, 1.8), self.data_set.new_sequence).start()

    def direction(self, degree):
        norm_degree = Utility.degree_norm(degree)*180/math.pi
        if Utility.in_threshold(norm_degree, 90, degree_delta):
            return self.dataset.get_right()

        elif Utility.in_threshold(norm_degree, -90, degree_delta):
            return self.dataset.get_left()

        elif Utility.in_threshold(norm_degree, 0, degree_delta):
            return self.dataset.get_forward()

        else:
            return "Turning around"

    def add_lang_str(self, lang_srt):
        self.lock.acquire()

        if self.lang_bag_str.data == "nothing":
            self.lang_bag_str.data = ""
        if  self.lang_bag_str.data != "":
            self.lang_bag_str.data += " and "
        self.lang_bag_str.data += lang_srt
        self.remaining_laser_scan = random.randint(0, 5)  # make sure we have enough data before saving new sequence
        self.not_change = 6

        self.lock.release()

    def check_next_intersection(self):
        if Utility.distance_vector(self.list_intersection[self.next_pose].pose, self.pose) < intersection_r:
            num_ways = len(self.list_intersection[self.next_pose].connection)
            eligible_neighbor = []
            for neighbor in self.list_intersection[self.next_pose].connection:
                if neighbor != self.prev_pose:
                    eligible_neighbor.append(neighbor)
            index = 0
            pose_intersection = self.list_intersection[self.next_pose].pose
            pose_prev = self.list_intersection[self.prev_pose].pose
            ab = pose_prev - pose_intersection
            str_msg = String()
            if self.prev_pose == self.next_pose:
                rospy.logerr("prev pose and next pose are equal")
            if num_ways == 1:
                str_msg.data = "Dead end"
            if num_ways == 2:
                str_msg.data = self.dataset.get_corner()
            elif num_ways == 3:
                str_msg.data = self.dataset.get_intersection()
                index = random.randint(0, 1)

            if (self.data_set_ver == 1):
                self.dataset.write_bag(self.language_topic, str_msg)
            elif(self.data_set_ver == 2):
                self.lang_bag_str= str_msg

            self.remaining_laser_scan = random.randint(1, 5)      # make sure we have enough data before saving new sequence
            self.not_change = 10

            bc = pose_intersection - self.list_intersection[eligible_neighbor[index]].pose
            degree = Utility.degree_vector(ab, bc)
            str_msg = String()
            str_msg.data = self.direction(degree)
            if (self.data_set_ver == 1):
                self.dataset.write_bag(self.language_topic, str_msg)
            elif(self.data_set_ver == 2):
                self.lang_bag_str.data = self.lang_bag_str.data + " " + str_msg.data

            self.prev_pose = self.next_pose
            self.next_pose = eligible_neighbor[index]
            self.send_pos(self.list_intersection[eligible_neighbor[index]].pose)
            # target_goal_simple = MoveBaseActionGoal()
            #
            # # forming a proper PoseStamped message
            # target_goal_simple.goal.target_pose.pose.position.x = \
            #     self.list_intersection[eligible_neighbor[index]].pose[0]
            # target_goal_simple.goal.target_pose.pose.position.y = \
            #     self.list_intersection[eligible_neighbor[index]].pose[1]
            # target_goal_simple.goal.target_pose.pose.position.z = 0
            # target_goal_simple.goal.target_pose.pose.orientation.w = 1
            # target_goal_simple.goal.target_pose.header.frame_id = 'map'
            # target_goal_simple.goal.target_pose.header.stamp = rospy.Time.now()
            # target_goal_simple.header.stamp = rospy.Time.now()
            # self.publisher.publish(target_goal_simple)
            self.remaining_intersection -= 1
            if self.remaining_intersection == 0 and self.data_set_ver ==1:
                self.reset_intersection_collection()
        else:
            if self.not_change <= 0:
                self.lang_bag_str.data = "nothing"
