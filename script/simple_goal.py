#! /usr/bin/env python

import roslib
# roslib.load_manifest('blob_follower')
import rospy
import actionlib
from mbf_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import PoseStamped

def callback(data):
    client = actionlib.SimpleActionClient("move_base_flex/move_base", MoveBaseAction)

    # client = actionlib.SimpleActionClient('do_dishes', DoDishesAction)
    client.wait_for_server()

    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = 'map'
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = data.pose.position.x
    goal.target_pose.pose.position.y = data.pose.position.y
    goal.target_pose.pose.orientation.w = 1  # go forward

    # start moving
    client.send_goal(goal)

    success = client.wait_for_result(rospy.Duration.from_sec(5.0))
    print('success status', success)



if __name__ == '__main__':
    rospy.init_node('move_base_flex_action_lib')

    rospy.Subscriber("/move_base_simple/goal", PoseStamped, callback)
    rospy.spin()
