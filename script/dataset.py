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
import thread
import constants
import rospkg
from utility import *
import pickle
import math
import os
import torch

from generate_map import  GenerateMap

import time

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
    def __init__(self, _dataset_directory=None, is_online=False):
        """

        :param _dataset_directory: directory containing pickle files
        :param is_online: are you using robot online data or saved pickle file
        """
        self.is_online = is_online
        self.generate_map = None
        self._dataset_directory = _dataset_directory
        if not self.is_online:
            self.files = [os.path.join(self._dataset_directory, f) for f in os.listdir(self._dataset_directory) if
                          os.path.isfile(os.path.join(self._dataset_directory, f))]
        else:

            self.generate_map = GenerateMap(is_online=is_online)
            self.generate_map.init_map()

            print ("after GenerateMap online")
            t = threading.Thread(target=self.generate_map.read_from_pickle, args = ())
            t.daemon = True
            t.start()
           
            print ("after read from pickle map dataset")

        self.word_encoding = model_utils.WordEncoding()



    def __len__(self):
        if self.is_online:
            return 100000
        return len(self.files)

    def __getitem__(self, i):
        if not self.is_online:
                try:
                    dic_data = pickle.load(f)
                except ValueError:
                    print (self.files[i])
                    exit(0)
                language = dic_data["language"]
                laser_scans = dic_data["laser_scan"]
                local_maps = dic_data["local_maps"]
        else:
            while not self.generate_map.new_data_ready:
                time.sleep(0.1)
                # print ("wait for data")

            print ("got a data")
            language, laser_scans, local_maps = self.generate_map.get_data()



        word_encoded = list(map(self.word_encoding.get_object_class, language))
        word_encoded = [(x[0], (x[1][0]/constants.LOCAL_MAP_DIM*constants.GRID_LENGTH, (x[1][1]/constants.LOCAL_MAP_DIM+0.5) * constants.GRID_LENGTH), x[2]) for x in word_encoded ]

        # width , height, two anchors, objectness + (x, y) + classes
        target = torch.zeros([constants.GRID_LENGTH, constants.GRID_LENGTH, 2, 4], dtype=torch.float)

        for word in word_encoded:

            # center of object in tile
            x_center = word[1][0]%1
            y_center = word[1][1]%1

            # x,y coordinate in gridded map
            x = int(math.floor(word[1][0]))
            y = int(math.floor(word[1][1]))
            if x >= constants.GRID_LENGTH:
                x = 4
                x_center = 0.99999
            if y >= constants.GRID_LENGTH:
                y = 4
                y_center = 0.99999


            # objectness between 0.4 to 1
            try:
                target[x][y][word[2]][0] = max(min((1-x_center) * 3, 1), 0.4)
            except Exception as e:
                print e
                print ("target shape: ", target.shape)
                print ("word: ", len(word))
                print ("x:, y:, (1-x_center) * 3:", x, y, (1 - x_center) * 3)
                exit(2)
            # x_center , y_center
            target[x][y][word[2]][1] = x_center
            target[x][y][word[2]][2] = y_center

            # class
            target[x][y][word[2]][3] = word[0]

        target_classes = target[ :, :, :, 3:]
        target_poses = target[ :, :, :, 1:3]
        target_objectness = target[ :, :, :, 0]

        laser_scan_map = model_utils.laser_to_map(laser_scans, constants.LASER_FOV, 240, constants.MAX_RANGE_LASER)
        laser_scans_map = torch.from_numpy(np.stack([laser_scan_map, laser_scan_map, laser_scan_map])).type(torch.FloatTensor)

        local_maps = cv2.resize(local_maps, (240, 240))

        laser_scan_map = laser_scan_map.squeeze(2)
        local_maps = torch.from_numpy(np.stack([local_maps, laser_scan_map, laser_scan_map])).type(torch.FloatTensor)

        # local_maps = torch.from_numpy(local_maps).type(torch .FloatTensor).unsqueeze(0)

        return target_classes.type(torch.LongTensor), target_poses, target_objectness, local_maps, laser_scans_map



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

