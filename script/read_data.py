
import sys

def fix_python_path():
    if sys.version_info.major == 3:
        sys.path = [i for i in sys.path if 'python2' not in i]
    elif sys.version_info.major == 2:
        sys.path = [i for i in sys.path if 'python3' not in i]

def add_current_path():
    print('adding path: ', __file__)
    sys.path.append(__file__)

#import rospy
# from robot import Robot
from model import Map_Model
#import rospkg
import os
from dataset import Laser_Dataset, Map_Dataset
import constants
from torch import nn
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.model_utils import WordEncoding
from torch.autograd import Variable
import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description='train, test or debug model')

    parser.add_argument(
        '--train', metavar='train', type=str,
        help='train dataset')
#    parser.set_defaults(train=os.path.join(rospkg.RosPack().get_path('robot_describe'), "data", "dataset", "train"))

    parser.add_argument(
        '--validation', metavar='validation', type=str,
        help='validation dataset')
 #   parser.set_defaults(validation=os.path.join(rospkg.RosPack().get_path('robot_describe'), "data", "dataset", "validation"))
    parser.add_argument(
        '--resume', metavar='resume', type=str,
        help='resume checkpoint')
   # parser.set_defaults(resume=os.path.join(rospkg.RosPack().get_path('robot_describe'),'checkpoints','checkpoint.pth.tar'))

    parser.add_argument(
        '--batchSize', metavar='batchSize', type=int, default=50,
        help='batch size')

    parser.add_argument(
        '--debug', dest='debug', action='store_true',
        help='is dubug')
    parser.set_defaults(debug=False)

    parser.add_argument(
        '--no-cuda', dest='isCuda', action='store_false',
        help='no cuda')
    parser.set_defaults(isCuda=True)


    args = parser.parse_args()

    if args.debug:
        print ("debug mode")

    # rospy.init_node('listener', anonymous=True)



    map_dataset_train = Map_Dataset(args.train, augmentation=True)
    map_dataset_validation = Map_Dataset(args.validation)

    my_model = Map_Model(map_dataset_train, map_dataset_validation,
                         args.resume,
                         save=True, load_weight=True, cuda=args.isCuda)

    # to debug:
    # my_model = Map_Model(map_dataset_train, map_dataset_validation,
    #                      args.resume,
    #                      save=False, load_weight=True, log=False, cuda=True)
    # my_model.visualize_dataset(args.batchSize, map_dataset_train)
    # exit(0)

    my_model.train_iters(10000, print_every=10, save=True, batch_size=args.batchSize, num_workers=4)
