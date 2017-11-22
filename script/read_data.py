import rospy
from robot import Robot
from train import *


if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    robot = Robot(False)
    my_model = Model(robot.data_set)