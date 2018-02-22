import rospy
from nav_msgs.msg import Odometry
# from move_base_msgs.msg import MoveBaseActionGoal
from mbf_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped

from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import random
from numpy import *
import threading
import rospkg
from dataset import *
from utility import *
import actionlib

class IntersectionNode:
    def __init__(self, pose, connected_nodes):
        self.connection = connected_nodes
        self.pose = pose


class Robot:
    def __init__(self, is_data_generation, data_set_ver, start_index_dataset = 0):
        random.seed()

        rospack = rospkg.RosPack()
        self.data_set_ver = data_set_ver
        self.remaining_intersection = 1
        self.path = rospack.get_path('blob_follower')
        self.language_topic = "language"
        self.lock = threading.Lock()
        self.not_change = 0
        if data_set_ver == 1:
            self.data_set = DataSet(self.path + "/data/directions.txt", self.path + "/bag/bag_", is_data_generation,bag_number = start_index_dataset)
        else:
            self.data_set = DataSet(self.path + "/data/directions.txt", self.path + "/bag_2/", is_data_generation, bag_number = start_index_dataset)
        self.lang_bag_str = String()
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
            self.remaining_laser_scan = 40  # start with 40 which means wait for 40 laser scan as the array empty
            self.laser_scans = [LaserScan() for x in range(40)]  # allocate 40 space for 40 laser scan
            self.laser_pointer = 0  # index to add new laser scan
            self.send_pos(self.list_intersection[0].pose)


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


    def save_bag_scan_laser(self, scan):
        self.lock.acquire()
        # success = self.client.get_goal_status_text()
        # print('success status', success)
        self.laser_scans[self.laser_pointer] = scan
        self.laser_pointer = (self.laser_pointer + 1) % 40

        self.remaining_laser_scan -= 1
        if (self.remaining_laser_scan <= 0):        # getting to limit so save the bag files
            if (self.lang_bag_str.data == "continue straight" and random.uniform(0, 1.0) > 0.9):
                self.remaining_laser_scan = 20
                self.lock.release()
                return
            rospy.logerr(self.lang_bag_str.data)
            self.data_set.write_bag(self.language_topic, self.lang_bag_str)
            self.remaining_laser_scan = 20
            for i in range (40):
                if (not self.data_set.write_bag("/robot_0/base_scan_1", self.laser_scans[(i + self.laser_pointer) % 40])):
                    self.not_change = 0
                    self.lock.release()
                    return

            threading.Timer(0.0, self.data_set.new_sequence).start()
            if self.not_change > 0:
                self.remaining_laser_scan = 2
                self.not_change -= 1
        self.lock.release()

    def reset_intersection_collection(self):
        # self.remaining_intersection = random.randint(3, 7)
        self.remaining_intersection = 1
        threading.Timer(0.5, self.data_set.new_sequence).start()
        # threading.Timer(random.uniform(0.5, 1.8), self.data_set.new_sequence).start()

    def direction(self, degree):
        norm_degree = Utility.degree_norm(degree)*180/math.pi
        if Utility.in_threshold(norm_degree, 90, degree_delta):
            return self.data_set.get_right()

        elif Utility.in_threshold(norm_degree, -90, degree_delta):
            return self.data_set.get_left()

        elif Utility.in_threshold(norm_degree, 0, degree_delta):
            return self.data_set.get_forward()

        else:
            return "Turning around"


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
                str_msg.data = self.data_set.get_corner()
            elif num_ways == 3:
                str_msg.data = self.data_set.get_intersection()
                index = random.randint(0, 1)

            if (self.data_set_ver == 1):
                self.data_set.write_bag(self.language_topic, str_msg)
            elif(self.data_set_ver == 2):
                self.lang_bag_str= str_msg

            self.remaining_laser_scan = random.randint(1, 5)      # make sure we have enough data before saving new sequence
            self.not_change = 10

            bc = pose_intersection - self.list_intersection[eligible_neighbor[index]].pose
            degree = Utility.degree_vector(ab, bc)
            str_msg = String()
            str_msg.data = self.direction(degree)
            if (self.data_set_ver == 1):
                self.data_set.write_bag(self.language_topic, str_msg)
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
                self.lang_bag_str.data = "continue straight"
