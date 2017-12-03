import random
import rosbag
from sensor_msgs.msg import LaserScan
import threading
import torch
import rospy
from torch.autograd import Variable

EOS_token = 1
use_cuda = torch.cuda.is_available()


class Data:
    def __init__(self):
        self.laser = []
        self.words = ""


class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS

    def add_words(self, words):
        for word in words:
            self.add_word(word)

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class DataSet:
    def __init__(self, directions, data_generation, is_data_generation):
        self._list_iterator = 0
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
            self._max_length_laser = 0
            self.lang = Lang()
            self._bag_num = 0 # means no bag read yet
            rospy.loginfo("number of bag read: " + self._bag_name + str(self._bag_num+1))
            self._bag = rosbag.Bag(self._bag_name + str(self._bag_num+1), 'r')
            self.read_bag()

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
                print (self._bag_name + str(self._bag_num+1))

                for topic, msg, t in bag.read_messages(topics=['language', '/robot_0/base_scan_1']):
                    if topic == 'language':
                        rospy.loginfo(msg)
                        data.words += msg.data
                    else:
                        counter += 1
                        # ranges = tuple(int(100000 * x) for x in msg.ranges)
                        data.laser.append([float(x)/msg.range_max for x in msg.ranges])
                self._bag_num += 1
                bag.close()
                if len(data.laser) > self._max_length_laser:
                    self._max_length_laser = len(data.laser)
                data.words = data.words.replace('\n', ' ')
                data.words = [a for a in data.words.split(' ') if a != '']
                self.lang.add_words(data.words)
                self.list_data.append(data)
            except IOError:
                rospy.loginfo("number of bag read: " + str(self._bag_num) + "laser: " +str(counter))
                break


    def new_sequence(self):
        rospy.logwarn("saving " + str(self._bag_name) + str(self._bag_num))
        self._bag.close()
        self._bag_num += 1
        self._bag = rosbag.Bag(self._bag_name + str(self._bag_num), 'w')

    def shuffle_data(self):
        random.shuffle(self.list_data)
        self._list_iterator = 0

    def next_pair(self):
        pair = self.list_data[self._list_iterator]
        self._list_iterator += 1
        words = [self.lang.word2index[word] for word in pair.words]
        words.append(EOS_token)
        words = Variable(torch.LongTensor(words).view( -1, 1))
        laser = Variable(torch.DoubleTensor(pair.laser).view( -1, len(pair.laser[0]))).float()

        if use_cuda:
            words = words.cuda()
            laser = laser.cuda()

        return laser, words





