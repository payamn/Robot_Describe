#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry
from move_base_msgs.msg import MoveBaseActionGoal
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import random
from numpy import *
import threading
import rosbag
import rospkg

mapper_constant_val_x = 10
mapper_constant_val_y = 10
intersection_r = 0.5
degree_delta = 25


class DataSet:

    def read_list_direction(self):
        list_temp = []
        line = self._directions.readline()
        while line != "-\n":
            list_temp.append(line)
            line = self._directions.readline()
            print line
        return list_temp

    def get_right(self):
        return random.choice(self._right)

    def get_left(self):
        return random.choice(self._left)

    def get_forward(self):
        return random.choice(self._forward)

    def get_corner(self):
        return random.choice(self._corner) + random.choice(self._one_way)

    def get_intersection(self):
        return random.choice(self._intersection) + random.choice(self._two_way)

    def read_directions(self):
        self._corner = self.read_list_direction()
        self._left = self.read_list_direction()
        self._right = self.read_list_direction()
        self._forward = self.read_list_direction()
        self._two_way = self.read_list_direction()
        self._one_way = self.read_list_direction()
        self._intersection = self.read_list_direction()

    def write_bag(self, topic, msg_data):
        self.lock.acquire()
        if type(msg_data) != LaserScan:
            rospy.logwarn(msg_data.data)
        self._bag.write(topic, msg_data)
        self.lock.release()

    def new_sequence(self):
        rospy.loginfo("saving " + str(self._bag_name) + str(self._bag_num))
        self._bag.close()
        self._bag_num += 1
        self._bag = rosbag.Bag(self._bag_name + str(self._bag_num), 'w')

    def __init__(self, directions, data_generation):
        self.lock = threading.Lock()
        self._directions = open(directions, 'r')
        self._bag_num = 0
        self._bag_name = data_generation
        self._bag = rosbag.Bag(self._bag_name + str(self._bag_num), 'w')
        self._corner = []
        self._left = []
        self._right = []
        self._forward = []
        self._two_way = []
        self._one_way = []
        self._intersection = []
        self.read_directions()
        print self._corner
        print self._left
        print self._right
        print self._forward
        print self._intersection


class Utility:

    @staticmethod
    def distance_vector(a, b):
        return math.sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]))

    @staticmethod
    def val_vector(a):
        return math.sqrt(a[0] * a[0] + a[1] * a[1])

    @staticmethod
    def dot_product(a, b):
        return a[0] * b[0] + a[1] * b[1]

    @staticmethod
    def degree_vector(a, b):
        return math.atan2(a[1], a[0]) - math.atan2(b[1], b[0])

    @staticmethod
    def degree_norm(degree):
        return math.atan2(math.sin(degree), math.cos(degree))

    @staticmethod
    def in_threshold(x, num, threshold):
        if math.fabs(x - num) < threshold:
            return True
        return False


class IntersectionNode:
    def __init__(self, pose, connected_nodes):
        self.connection = connected_nodes
        self.pose = pose


class Robot:
    def __init__(self):
        random.seed()

        # get an instance of RosPack with the default search paths
        rospack = rospkg.RosPack()

        # get the file path for rospy_tutorials
        self.remaining_intersection = random.randint(3, 7)
        self.path = rospack.get_path('blob_follower')
        self.language_topic = "language"
        self.lock = threading.Lock()
        self.data_set = DataSet(self.path + "/data/directions.txt", self.path + "/bag/bag_")
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
        self.publisher = rospy.Publisher('/move_base/goal', MoveBaseActionGoal, queue_size=10, latch=True)
        target_goal_simple = MoveBaseActionGoal()

        # publish a goal for move base to navigate to intersection 0 for start
        target_goal_simple.goal.target_pose.pose.position.x = \
            self.list_intersection[0].pose[0]
        target_goal_simple.goal.target_pose.pose.position.y = \
            self.list_intersection[0].pose[1]
        target_goal_simple.goal.target_pose.pose.position.z = 0
        target_goal_simple.goal.target_pose.pose.orientation.w = 1
        target_goal_simple.goal.target_pose.header.frame_id = 'map'
        target_goal_simple.goal.target_pose.header.stamp = rospy.Time.now()
        target_goal_simple.header.stamp = rospy.Time.now()
        self.publisher.publish(target_goal_simple)

    def direction(self, degree):
        norm_degree = Utility.degree_norm(degree)*180/math.pi
        rospy.logerr(norm_degree)
        if Utility.in_threshold(norm_degree, 90, degree_delta):
            return self.data_set.get_right()

        elif Utility.in_threshold(norm_degree, -90, degree_delta):
            return self.data_set.get_left()

        elif Utility.in_threshold(norm_degree, 0, degree_delta):
            return self.data_set.get_forward()

        else:
            return "Turning around"

    def reset_intersection_collection(self):
        self.remaining_intersection = random.randint(3, 7)
        threading.Timer(random.uniform(0.5, 4.1), self.data_set.new_sequence).start()


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
                index = random.randint(0, 2)

            self.data_set.write_bag(self.language_topic, str_msg)
            bc = pose_intersection - self.list_intersection[eligible_neighbor[index]].pose
            degree = Utility.degree_vector(ab, bc)
            str_msg = String()
            str_msg.data = self.direction(degree)
            self.data_set.write_bag(self.language_topic, str_msg)

            self.prev_pose = self.next_pose
            self.next_pose = eligible_neighbor[index]

            target_goal_simple = MoveBaseActionGoal()

            # forming a proper PoseStamped message
            target_goal_simple.goal.target_pose.pose.position.x = \
                self.list_intersection[eligible_neighbor[index]].pose[0]
            target_goal_simple.goal.target_pose.pose.position.y = \
                self.list_intersection[eligible_neighbor[index]].pose[1]
            target_goal_simple.goal.target_pose.pose.position.z = 0
            target_goal_simple.goal.target_pose.pose.orientation.w = 1
            target_goal_simple.goal.target_pose.header.frame_id = 'map'
            target_goal_simple.goal.target_pose.header.stamp = rospy.Time.now()
            target_goal_simple.header.stamp = rospy.Time.now()
            self.publisher.publish(target_goal_simple)
            self.remaining_intersection -= 1
            if self.remaining_intersection == 0:
                self.reset_intersection_collection()


def callback_laser_scan(scan, my_robot):
    my_robot.data_set.write_bag("/robot_0/base_scan_1", scan)


def callback_robot_0(odom_data, my_robot):
    pose = odom_data.pose.pose.position
    my_robot.pose = array([pose.x+mapper_constant_val_x, pose.y + mapper_constant_val_y])
    my_robot.check_next_intersection()

if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    robot = Robot()
    rospy.Subscriber("/robot_0/base_pose_ground_truth", Odometry, callback_robot_0, robot)
    rospy.Subscriber("/robot_0/base_scan_1", LaserScan, callback_laser_scan, robot)
    rospy.spin()
