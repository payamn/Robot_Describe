#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped

from move_base_msgs.msg import MoveBaseActionGoal
# from numpy.doc.constants import prev
from sensor_msgs.msg import LaserScan
import cv2

from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from std_msgs.msg import String
import random
import tf
from tf import TransformListener
import numpy as np
import threading
import rospkg
from utility import *
import pickle
import math
import os
import torch

from utils import model_utils

class Laser_Dataset(Dataset):
    def __init__(self, _dataset_directory, prediction_number = 5):
        self._dataset_directory = _dataset_directory
        self.files = [os.path.join(self._dataset_directory, f) for f in os.listdir(self._dataset_directory) if
                 os.path.isfile(os.path.join(self._dataset_directory, f))]
        self.prediction_number = prediction_number
        self.laser_window = 20
        self.word_encoding = model_utils.WordEncoding()
    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with open(self.files[i], 'rb') as f:
            dic_data = pickle.load(f)

            language = dic_data["language"] if len(dic_data["language"])>0 else [(-1,"noting",(0,0))]
            word_encoded = list(map(self.word_encoding.get_object_class, language))
            for index in range(self.prediction_number-len(word_encoded)):
                word_encoded.append((self.word_encoding.classes["noting"],(0,0)))
            word_encoded_class, word_encoded_pose= zip(*word_encoded)
            laser_scans = dic_data["laser_scans"]
            speeds = dic_data["speeds"]
            laser_scans = torch.from_numpy(np.stack(laser_scans)).type(torch.FloatTensor)
            speeds = torch.from_numpy(np.stack(speeds)).type(torch.FloatTensor)
            word_encoded_pose = torch.FloatTensor(word_encoded_pose)
            word_encoded_class = torch.LongTensor(word_encoded_class)
            return word_encoded_class, word_encoded_pose, speeds,laser_scans

class Map_Dataset(Dataset):
    def __init__(self, _dataset_directory, prediction_number=5):
        self._dataset_directory = _dataset_directory
        self.files = [os.path.join(self._dataset_directory, f) for f in os.listdir(self._dataset_directory) if
                      os.path.isfile(os.path.join(self._dataset_directory, f))]
        self.prediction_number = prediction_number
        self.word_encoding = model_utils.WordEncoding()



    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with open(self.files[i], 'rb') as f:
            try:
                dic_data = pickle.load(f)
            except ValueError:
                print (self.files[i])
                exit(0)
            language = dic_data["language"] if len(dic_data["language"]) > 0 else [(-1, "noting", (0, 0))]
            word_encoded = list(map(self.word_encoding.get_object_class, language))
            for index in range(self.prediction_number - len(word_encoded)):
                word_encoded.append((self.word_encoding.classes["noting"], (0, 0)))
            word_encoded_class, word_encoded_pose = zip(*word_encoded)
            local_maps = dic_data["local_maps"]
            local_maps = cv2.resize(local_maps, (224, 224))
            local_maps = torch.from_numpy(np.stack([local_maps, local_maps, local_maps])).type(torch.FloatTensor)
            word_encoded_pose = [x for x in enumerate(word_encoded_pose)]
            word_encoded_pose.sort(key=lambda l: (l[1][0], l[1][1]))
            word_encoded_class = [word_encoded_class[word_encoded_pose[x][0]] for x in range(len(word_encoded_class))]
            #word_encoded_class_parent = [self.word_encoding.get_parent_class(word_class) for word_class in word_encoded_class]
            word_encoded_pose = [x[1] for x in word_encoded_pose]
            word_encoded_pose = torch.FloatTensor(word_encoded_pose)
            word_encoded_class = torch.LongTensor(word_encoded_class)
            #word_encoded_class_parent = torch.LongTensor(word_encoded_class_parent)
        return word_encoded_class, word_encoded_pose, local_maps



# if __name__ == '__main__':
#     import argparse
#
#     parser = argparse.ArgumentParser(description='dataset')
#     # parser.add_argument('--generate_point', dest='generate_point', action='store_true')
#     parser.add_argument('--dataset', type=str, default="data/dataset", help='FOO!')
#     # parser.set_defaults(generate_point=False)
#     args = parser.parse_args()
#
#     generator = Map_Dataset(args.dataset)
#     dataloader = DataLoader(generator, shuffle=True, num_workers=10, batch_size= 10)
#     for i, ( word_encoded_class, word_encoded_pose, local_maps) in enumerate(dataloader):
#
#         print('label: ', word_encoded_class.size())
#         print('input: ', speeds.size())

