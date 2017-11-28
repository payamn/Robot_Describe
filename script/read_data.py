import rospy
from robot import Robot
from train import *
import rospkg

if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    robot = Robot(False)
    my_model = Model(robot.data_set, rospkg.RosPack().get_path('blob_follower') + "/check_points/",
                     resume_path=rospkg.RosPack().get_path('blob_follower') + "/check_points/" +"_best_" )