#!/usr/bin/env python
import rospy
from nav_msgs.msg import Odometry, OccupancyGrid
from geometry_msgs.msg import PoseStamped

from move_base_msgs.msg import MoveBaseActionGoal
# from numpy.doc.constants import prev
from sensor_msgs.msg import LaserScan
import cv2


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

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with open(self.files[i], 'rb') as f:
            dic_data = pickle.load(f)
            word_encoding = model_utils.WordEncoding()
            language = dic_data["language"] if len(dic_data["language"])>0 else [(-1,"noting",(0,0))]
            word_encoded = list(map(word_encoding.get_object_class, language))
            word_encoded_pose, word_encoded_class = zip(*word_encoded)
            laser_scans = dic_data["laser_scans"]
            speeds = dic_data["speeds"]
            laser_scans = torch.from_numpy(np.stack(laser_scans))
            speeds = torch.from_numpy(np.stack(speeds))
            word_encoded_pose = torch.LongTensor(word_encoded_pose)
            word_encoded_class = torch.FloatTensor(word_encoded_class)
            return word_encoded_class, word_encoded_pose, speeds,laser_scans



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='dataset')
    # parser.add_argument('--generate_point', dest='generate_point', action='store_true')
    parser.add_argument('--dataset', type=str, default="data/dataset", help='FOO!')
    # parser.set_defaults(generate_point=False)
    args = parser.parse_args()

    generator = Laser_Dataset(args.dataset)
    dataloader = DataLoader(generator, shuffle=True, num_workers=10)
    for i, ( word_encoded_class, word_encoded_pose, speeds,laser_scans) in enumerate(dataloader):

        print('label: ', word_encoded_class.size())
        print('input: ', speeds.size())

