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
    robot = Robot(False)
    my_model = Model(robot.data_set, rospkg.RosPack().get_path('blob_follower') + "/check_points/",
                     resume_path=rospkg.RosPack().get_path('blob_follower') + "/check_points/" +"_best_" ,teacher_forcing_ratio=0.5)
