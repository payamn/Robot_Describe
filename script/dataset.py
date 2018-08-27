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
    def __init__(self, _dataset_directory=None, augmentation=False):
        """

        :param _dataset_directory: directory containing pickle files
        :param is_online: are you using robot online data or saved pickle file
        """
        self.generate_map = None
        self._dataset_directory = _dataset_directory
        self.files = [os.path.join(self._dataset_directory, f) for f in os.listdir(self._dataset_directory) if
                      os.path.isfile(os.path.join(self._dataset_directory, f))]

        self.word_encoding = model_utils.WordEncoding()
        self.augmentation = augmentation
        self.augmentation_level = 0
        print (_dataset_directory, "augmentation: ", augmentation)

    def set_augmentation_level(self, number):
        if number != self.augmentation_level:
            self.augmentation_level = number
            print ("augmentation level changed to: ", self.augmentation_level)



    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with open(self.files[i], 'rb') as f:
            try:
                dic_data = pickle.load(f)
            except ValueError:
                print (self.files[i])
                exit(0)
        language = dic_data["language"]
        laser_scans = dic_data["laser_scan"]
        local_maps = dic_data["local_maps"]

        angle = None
        transform = None
        resize = None
        if self.augmentation:
            if self.augmentation_level > 0:
                # max transform pose is +0.2 out of 1
                transform = (random.randint(-2000, 0) / 10000.0, random.randint(-2000, 2000) / 10000.0)

            if self.augmentation_level > 1:
                # max angle transform -+30 degree
                angle = random.randint(-300, 300)/10.0

            if self.augmentation_level > 2:
                # resize between 0.85x to 1.15x
                resize = random.randint(85,115)/100.0


        word_encoded = []
        words = list(map(self.word_encoding.get_object_class, language))
        center = (0, constants.LOCAL_DISTANCE/ 2)
        for word in words:
            x, y = word[1]
            y = y + constants.LOCAL_DISTANCE/ 2
            if resize:
                # x = x * resize + (1 - resize) * 0.5 * constants.LOCAL_DISTANCE
                x = x * resize
                y = y * resize + (1 - resize) * 0.5 * constants.LOCAL_DISTANCE

            if angle:
                x, y = Utility.rotate_point(center,(x, y), math.radians(-angle))
            x = x / constants.LOCAL_MAP_DIM
            y = (y - constants.LOCAL_DISTANCE/2) / (constants.LOCAL_MAP_DIM) + 0.5
            if transform:
                x += transform[0]
                y += transform[1]

            if 0.1 < x < 0.9 and 0.1 < y < 0.9:
                x = x * constants.GRID_LENGTH
                y = y * constants.GRID_LENGTH
                word_encoded.append((word[0], (x, y), word[2], word[3], angle, transform))


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


            try:
                target[x][y][word[2]][0] = word[3]
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

        laser_scan_map = model_utils.laser_to_map(laser_scans, constants.LASER_FOV, 240, constants.MAX_RANGE_LASER, angle, transform, resize=resize)
        laser_scan_map_low = model_utils.laser_to_map(laser_scans, constants.LASER_FOV, 240, constants.MAX_RANGE_LASER, angle, transform, resize=resize, circle_size=1)
        laser_scans_map = torch.from_numpy(np.stack([laser_scan_map, laser_scan_map, laser_scan_map])).type(torch.FloatTensor)

        local_maps = Utility.sub_image(local_maps, 0.0500000007451, center, angle, constants.LOCAL_MAP_DIM,
                                       constants.LOCAL_MAP_DIM, only_forward=True, transform=transform, resize=resize
                                       )
        local_maps = cv2.resize(local_maps, (240, 240))

        laser_scan_map = laser_scan_map.squeeze(2)
        laser_scan_map_low = laser_scan_map_low.squeeze(2)
        local_maps = torch.from_numpy(np.stack([local_maps, laser_scan_map_low, laser_scan_map])).type(torch.FloatTensor)

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

