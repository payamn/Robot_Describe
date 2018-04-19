import random
import rosbag
from sensor_msgs.msg import LaserScan
import threading
import torch
import rospy
import os
import cPickle
from torch.autograd import Variable

EOS_token = 1
use_cuda = torch.cuda.is_available()


class Data:
    def __init__(self):
        self.speed = []
        self.laser = []
        self.words = ""


class Lang:
    def __init__(self):
        self.word2index = {}
        self.word2count = {}
        self.index2word = {}
        # self.index2word = {0: "SOS", 1: "EOS"}
        # Count SOS and EOS if it is language
        self.n_words = 0

    def reset_count(self):
        for word in self.word2index:
            self.word2count[word] = 0

    def add_words(self, words):
        for word in words:
            self.add_word(word)

    def add_classes(self, sentences):
        sentences.sort()
        sentences = " and ".join(sentences)
        self.add_word(sentences)
        return sentences

    def add_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1


class DataSet:
    def __init__(self, directions, data_generation, is_data_generation, batch_size = 1, bag_number = 0, use_direction_file=True, read_lang_disk=True):
        self._list_iterator = 0
        self.lock = threading.Lock()
        self.list_data = []
        self.batch_size = batch_size
        if (use_direction_file):
            self._directions = open(directions, 'r')
            self._corner = []
            self._left = []
            self._right = []
            self._forward = []
            self._two_way = []
            self._one_way = []
            self._intersection = []
        self._bag_num = bag_number
        self._bag_name = data_generation


        if is_data_generation:
            self._bag = rosbag.Bag(self._bag_name + str(self._bag_num), 'w')
            if (use_direction_file):
                self.read_directions()
                print self._corner
                print self._left
                print self._right
                print self._forward
                print self._intersection
        else:
            self._max_length_laser = 0
            self.lang = Lang()
            if read_lang_disk:
                self.load("data/lang.data")
                self.lang.reset_count()
            self._bag_num = bag_number # means no bag read yet
            rospy.loginfo("number of bag read: " + self._bag_name + str(self._bag_num+1))
            # self._bag = rosbag.Bag(self._bag_name + str(self._bag_num+1), 'r')
            self.read_bag()
            self.save("data/lang.data")

    def load(self, filename):
        f = open(filename, 'rb')
        self.lang = cPickle.load(f)
        f.close()

    def save(self, filename):
        f = open(filename, 'wb')
        cPickle.dump(self.lang, f, 2)
        f.close()

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

    # return True if laser was correct
    # return False if laser was not set correctly

    def write_bag(self, topic, msg_data):
        self.lock.acquire()
        # if type(msg_data) != LaserScan:
        #     rospy.logwarn("all laser are not set")
        #     self.lock.release()
        #     return False

        self._bag.write(topic, msg_data)
        self.lock.release()
        return True

    def read_bag(self):
        # rospy.logwarn("inside read bag: ")
        counter = 0
        files = [f for f in os.listdir(self._bag_name) if os.path.isfile(os.path.join(self._bag_name, f))]

        for file in files:
            data = Data()
            try:
                bag = rosbag.Bag(os.path.join(self._bag_name ,file) , 'r')
            except rosbag.bag.ROSBagUnindexedException:
                rospy.logerr("bag %s problem skipping:",file)
                continue

            # rospy.logwarn("inside read bag: " + str(self._bag_num))
            # rospy.logwarn( os.path.join(self._bag_name ,file))
            is_fail = False
            for topic, msg, t in bag.read_messages(topics=['language', '/robot_0/base_scan_1', '/robot_0/speed']):
                if topic == 'language':
                    # rospy.logwarn(msg)
                    data.words += msg.data
                elif topic == '/robot_0/speed':
                    data.speed.append([msg.linear.x, msg.linear.y, msg.angular.z])
                else:
                    data.laser.append([float(x) / msg.range_max for x in msg.ranges])
                    if len(data.laser[0])==0:
                        rospy.logerr("file without laser data:%s skipping", file)
                        is_fail = True
                        break
                    counter += 1
                    # ranges = tuple(int(100000 * x) for x in msg.ranges)
            if is_fail:
                continue
            if self._bag_num%100==0:
                rospy.loginfo("read %d bags",self._bag_num)
            self._bag_num += 1
            bag.close()
            if len(data.laser) > self._max_length_laser:
                self._max_length_laser = len(data.laser)
            data.words = data.words.replace('\n', ' ')

            # We can add words as classes:
            # data.words = [a for a in data.words.split(' ') if a != '']
            # self.lang.add_words(data.words)

            # We can add each class as our classes:
            data.words = [a for a in data.words.split("and") if a != '']
            data.sentence = self.lang.add_classes(data.words)

            self.list_data.append(data)
        rospy.logwarn("number of bag read: " + str(self._bag_num) + "laser: " + str(counter))
    # while True:
    #         try:
    #
    #
    #
    #
    #
    #         except IOError:
    #             rospy.loginfo("number of bag read: " + str(self._bag_num) + "laser: " +str(counter))
    #             break


    def new_sequence(self):
        self.lock.acquire()
        rospy.logwarn("saving " + str(self._bag_name) + str(self._bag_num))
        self._bag.close()
        self._bag_num += 1
        self._bag = rosbag.Bag(self._bag_name + str(self._bag_num), 'w')
        self.lock.release()


    def shuffle_data(self):
        random.shuffle(self.list_data)
        self._list_iterator = 0

    def next_batch(self):
        words_batch = []
        laser_batch = []
        speed_batch = []
        for batch in range(self.batch_size):
            laser, speed, words = self.next_pair()
            words_batch.append(words)
            laser_batch.append(laser)
            speed_batch.append(speed)
        return torch.stack(laser_batch), torch.stack(speed_batch), torch.stack(words_batch)

        return
    def next_pair(self):
        pair = self.list_data[self._list_iterator]
        self._list_iterator += 1

        # words = [self.lang.word2index[word] for word in pair.words]
        # words.append(EOS_token)
        object_class = self.lang.word2index[pair.sentence]
        # words = Variable(torch.LongTensor(words).view( -1, 1))
        words = Variable(torch.LongTensor([object_class]))
        laser = Variable(torch.DoubleTensor(pair.laser).view( -1, len(pair.laser[0]))).float()
        speed = Variable(torch.DoubleTensor(pair.speed).view( -1, len(pair.speed[0]))).float()

        if use_cuda:
            words = words.cuda()
            laser = laser.cuda()
            speed = speed.cuda()

        return laser, speed, words





