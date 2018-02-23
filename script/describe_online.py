import rospy
from robot import Robot
from train import *
from sensor_msgs.msg import LaserScan
import rospkg


def callback_laser_scan(scan, my_robot):
    my_robot.bag_scan_laser_online(scan)

# def callback_robot_0(odom_data, my_robot):
#     pose = odom_data.pose.pose.position
#     my_robot.pose = array([pose.x+mapper_constant_val_x, pose.y + mapper_constant_val_y])
#     my_robot.check_next_intersection()

import subprocess
try:
    rospy.get_master().getPid()
except:
    roscore = subprocess.Popen('roscore')
if __name__ == '__main__':
    rospy.init_node('describe', anonymous=True)
    robot = Robot(False, 2, )
    rospy.Subscriber("/robot_0/base_scan_1", LaserScan, callback_laser_scan, robot)

    my_model = Model(robot, rospkg.RosPack().get_path('blob_follower') + "/check_points_2/",
                     resume_path=rospkg.RosPack().get_path('blob_follower') + "/check_points_2/" +"_best_" ,
                     teacher_forcing_ratio=0.5, model_ver=2, save=False, real_time_test=True)
