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

    # rospy.init_node('listener', anonymous=True)
    map_dataset_train = Map_Dataset(os.path.join(rospkg.RosPack().get_path('robot_describe'), "script", "data", "dataset", "train"))
    map_dataset_validation = Map_Dataset(os.path.join(rospkg.RosPack().get_path('robot_describe'), "script", "data", "dataset", "test"))
    # dataloader_validation = DataLoader(map_dataset_validation, shuffle=True, num_workers=10, batch_size=1,
    #                                         drop_last=True)

    # word_encoding = WordEncoding()
    # for batch, (word_encoded_class, word_encoded_pose, local_map) in enumerate(dataloader_validation):
    #     local_map = Variable(local_map)
    #     word_encoding.visualize_map(local_map.data.numpy(),word_encoded_class[0], word_encoded_pose[0])
    # exit(0)
    my_model = Map_Model(map_dataset_train, map_dataset_validation,
                     resume_path=os.path.join(rospkg.RosPack().get_path('robot_describe'), "check_points/model_best.pth.tar") ,
                     save=True, number_of_iter=1000, batch_size=100)
