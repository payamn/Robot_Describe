import rospy
from nav_msgs.msg import Odometry
from move_base_msgs.msg import MoveBaseActionGoal
from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
import random
from numpy import *
import threading
import rospkg
from dataset import *
from utility import *

class IntersectionNode:
    def __init__(self, pose, connected_nodes):
        self.connection = connected_nodes
        self.pose = pose


class Robot:
    def __init__(self, is_data_generation):
        random.seed()

        # get an instance of RosPack with the default search paths
        rospack = rospkg.RosPack()

        # get the file path for rospy_tutorials
        self.remaining_intersection = random.randint(3, 7)
        self.path = rospack.get_path('blob_follower')
        self.language_topic = "language"
        self.lock = threading.Lock()
        self.data_set = DataSet(self.path + "/data/directions.txt", self.path + "/bag/bag_", is_data_generation)
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

    def reset_intersection_collection(self):
        self.remaining_intersection = random.randint(3, 7)
        threading.Timer(random.uniform(0.5, 4.1), self.data_set.new_sequence).start()

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
                index = random.randint(0, 1)

            self.data_set.write_bag(self.language_topic, str_msg)
            rospy.logerr(len(eligible_neighbor))
            rospy.logerr(len(self.list_intersection))
            rospy.logerr(index)

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