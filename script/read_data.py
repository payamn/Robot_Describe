import rospy
from robot import Robot


if __name__ == '__main__':
    rospy.init_node('listener', anonymous=True)
    robot = Robot(False)