import rospy
from robot import Robot
from model import *
import rospkg
import os

import subprocess
try:
    rospy.get_master().getPid()
except:
    roscore = subprocess.Popen('roscore')
if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    robot = Robot(False, 2, batch_size=100, use_direction_file=False, bag_name="/bag_train/")
    my_model = Model(robot,
                     resume_path=os.path.join(rospkg.RosPack().get_path('robot_describe'), "check_points/model_best.pth.tar") ,
                     save=True, number_of_iter=1000)
