import rospy
# from robot import Robot
from model import Map_Model
import rospkg
import os
from dataset import Laser_Dataset, Map_Dataset
from torch import nn
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from utils.model_utils import WordEncoding
from torch.autograd import Variable

if __name__ == '__main__':
    batch_size = 100
    # rospy.init_node('listener', anonymous=True)
    map_dataset_train = Map_Dataset(os.path.join(rospkg.RosPack().get_path('robot_describe'), "script", "data", "dataset", "train"))
    map_dataset_validation = Map_Dataset(os.path.join(rospkg.RosPack().get_path('robot_describe'), "script", "data", "dataset", "test"))

    my_model = Map_Model(map_dataset_train, map_dataset_validation,
                     resume_path=os.path.join(rospkg.RosPack().get_path('robot_describe'), "check_points/model_best.pth.tar") ,
                     save=True)
    # my_model.visualize_dataset("test")
    # exit(0)

    my_model.train_iters(1000, print_every=10, save=True, batch_size=batch_size)
