import rospy
# from robot import Robot
from model import Model, Map_Model
import rospkg
import os
from dataset import Laser_Dataset, Map_Dataset
from torch import nn
import torch
from torch.autograd import Variable
#
# import subprocess
# try:
#     rospy.get_master().getPid()
# except:
#     roscore = subprocess.Popen('roscore')
if __name__ == '__main__':

    # rospy.init_node('listener', anonymous=True)
    map_dataset_train = Map_Dataset(os.path.join(rospkg.RosPack().get_path('robot_describe'), "script", "data", "dataset", "train"))
    map_dataset_validation = Map_Dataset(os.path.join(rospkg.RosPack().get_path('robot_describe'), "script", "data", "dataset", "test"))
    my_model = Map_Model(map_dataset_train, map_dataset_validation,
                     resume_path=os.path.join(rospkg.RosPack().get_path('robot_describe'), "check_points/model_best.pth.tar") ,
                     save=True, number_of_iter=1000, batch_size=20)
