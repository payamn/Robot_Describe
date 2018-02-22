import rospy
from robot import Robot
from train import *
import rospkg

import subprocess
try:
    rospy.get_master().getPid()
except:
    roscore = subprocess.Popen('roscore')
if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    robot = Robot(False, 2)
    my_model = Model(robot.data_set, rospkg.RosPack().get_path('blob_follower') + "/check_points_2/",
                     resume_path=rospkg.RosPack().get_path('blob_follower') + "/check_points_2/" +"_best_" ,
                     teacher_forcing_ratio=0.5, model_ver=2, save=True, number_of_iter=100)
