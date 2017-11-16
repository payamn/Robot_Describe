import random
import rosbag
from sensor_msgs.msg import LaserScan
import threading
import rospy


class Data:
    def __init__(self):
        self.laser = []
        self.sentences = ""



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

    def read_bag(self):
        rospy.logwarn("inside read bag: ")
        counter = 0
        while True:
            try:
                data = Data()
                bag = rosbag.Bag(self._bag_name + str(self._bag_num+1), 'r')

                rospy.logwarn("inside read bag: " + str(self._bag_num+1))
                for topic, msg, t in bag.read_messages(topics=['language', '/robot_0/base_scan_1']):
                    if topic == 'language':
                        rospy.loginfo(msg)
                        data.sentences += msg.data
                    else:
                        counter += 1
                        data.laser.append(msg.ranges)
                self._bag_num += 1
                bag.close()
                self.list_data.append(data)
            except IOError:
                rospy.loginfo("number of bag read: " + str(self._bag_num) + "laser: " +str(counter))
                break


    def new_sequence(self):
        rospy.loginfo("saving " + str(self._bag_name) + str(self._bag_num))
        self._bag.close()
        self._bag_num += 1
        self._bag = rosbag.Bag(self._bag_name + str(self._bag_num), 'w')

    def __init__(self, directions, data_generation, is_data_generation):
        self.lock = threading.Lock()
        self.list_data = []
        self._directions = open(directions, 'r')
        self._bag_num = 0
        self._bag_name = data_generation
        self._corner = []
        self._left = []
        self._right = []
        self._forward = []
        self._two_way = []
        self._one_way = []
        self._intersection = []
        if is_data_generation:
            self._bag = rosbag.Bag(self._bag_name + str(self._bag_num), 'w')
            self.read_directions()
            print self._corner
            print self._left
            print self._right
            print self._forward
            print self._intersection
        else:
            self._bag_num = 0 # means no bag read yet
            rospy.loginfo("number of bag read: " + self._bag_name + str(self._bag_num+1))
            self._bag = rosbag.Bag(self._bag_name + str(self._bag_num+1), 'r')
            self.read_bag()



