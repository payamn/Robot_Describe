#!/usr/bin/env python
import rospy
import smach
import smach_ros
from mbf_msgs.msg import ExePathAction
from mbf_msgs.msg import ExePathResult

from mbf_msgs.msg import GetPathAction
from mbf_msgs.msg import GetPathResult

from mbf_msgs.msg import RecoveryAction
from mbf_msgs.msg import RecoveryResult


from geometry_msgs.msg import PoseStamped

from utility import Utility

from wait_for_goal import WaitForGoal
import threading
import time
import functools



@smach.cb_interface(input_keys=['target_pose', 'start_pose'])
def get_path_goal_cb(userdata, goal):
    # print ("in get path")
    goal.use_start_pose = True
    goal.tolerance = 0.2

    position, quaternion = Utility.get_robot_pose("/map_server")
    # print "start:", position, quaternion

    goal.target_pose = userdata.target_pose

    goal.start_pose = PoseStamped()
    goal.start_pose.header = goal.target_pose.header
    goal.start_pose.pose.position.x = position[0]
    goal.start_pose.pose.position.y = position[1]
    goal.start_pose.pose.orientation.z = quaternion[2]
    goal.start_pose.pose.orientation.w = quaternion[3]

    goal.planner = 'planner'
    # print "goal is: ", goal


@smach.cb_interface(
    output_keys=['outcome', 'message', 'path'],
    outcomes=['succeeded', 'failure'])
def get_path_result_cb(userdata, status, result):
    userdata.message = result.message
    userdata.outcome = result.outcome
    userdata.path = result.path
    if result.outcome == GetPathResult.SUCCESS:
        return 'succeeded'
    else:
        return 'failure'


@smach.cb_interface(input_keys=['path'])
def ex_path_goal_cb(userdata, goal):
    goal.path = userdata.path
    goal.controller = 'controller'


@smach.cb_interface(
    output_keys=['outcome', 'message', 'final_pose', 'dist_to_goal'],
    outcomes=['succeeded', 'failure'])
def ex_path_result_cb(userdata, status, result):
    userdata.message = result.message
    userdata.outcome = result.outcome
    userdata.dist_to_goal = result.dist_to_goal
    userdata.final_pose = result.final_pose
    if result.outcome == ExePathResult.SUCCESS:
        return 'succeeded'
    else:
        return 'failure'


@smach.cb_interface(input_keys=['recovery_flag'], output_keys=['recovery_flag'])
def recovery_goal_cb(userdata, goal):
    # TODO implement a more clever way to call the right behavior
    if not userdata.recovery_flag:
        goal.behavior = 'clear_costmap'
        userdata.recovery_flag = True
    else:
        goal.behavior = 'rotate_recovery'
        userdata.recovery_flag = False


@smach.cb_interface(
    output_keys=['outcome', 'message'],
    outcomes=['succeeded', 'failure'])
def recovery_result_cb(userdata, status, result):
    if result.outcome == RecoveryResult.SUCCESS:
        return 'succeeded'
    else:
        return 'failure'

class ExecPath(smach.State):

    def __init__(self):
        smach.State.__init__(self,
                             outcomes=['succeeded', 'preempted', 'failure', 'new_goal'],
                             input_keys=['start_pose', 'waypoints', 'tolerance', 'use_start_pose', 'path'],
                             output_keys=['start_pose', 'target_pose', 'waypoints', 'tolerance', 'use_start_pose', 'outcome', 'message', 'final_pose', 'dist_to_goal'])
        self.global_target_pose = PoseStamped()

        self.subscriber = None
        self.flag = None
        self.outcome = None

        self.subscriber = rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)

        self.flag_lock = threading.Lock()

    def set_simple_action_state(self, simple_action_state):
        self.simple_action_state = simple_action_state

    def set_flag(self, flag):
        self.flag_lock.acquire()
        self.flag = flag
        self.flag_lock.release()

    def get_flag(self):
        self.flag_lock.acquire()
        flag = self.flag
        self.flag_lock.release()
        return flag

    def execute(self, userdata):
        self.flag = None
        rate = 0.3

        # print(userdata)

        action_thread = threading.Thread(target=smach_ros.SimpleActionState.execute, args=(self.simple_action_state, userdata,))
        try:
            action_thread.start()
        except Exception as e:
            rospy.logerr('Exception in thread start' + e)

        # action_thread.join()

        while not rospy.is_shutdown():
            flag = self.get_flag()
            # print ("flag is :" + str(flag))
            if (flag is None or flag == 'ex_path_goal_cb') and not rospy.is_shutdown():
                time.sleep(rate)
                continue
            else:
                break

        transition = None
        if flag == 'goal_callback':
            # TODO: cancel the previous goal
            userdata.target_pose = self.global_target_pose
            # print "Target Pose:", self.global_target_pose.pose.position.x, self.global_target_pose.pose.position.y,\
            #       self.global_target_pose.pose.position.z
            if rospy.is_shutdown():
              transition = 'preempted'

            transition = 'new_goal'
        elif flag == 'ex_path_result_cb':
            rospy.logerr('End of exec' + self.outcome)
            transition = self.outcome
        else:
            rospy.logerr('Unknown state')
            transition = 'failure'

        self.set_flag(None)
        # self.subscriber.unregister()

        return transition

    def goal_callback(self, msg):
        # rospy.logerr("Received goal:")
        self.global_target_pose = msg
        flag = self.get_flag()
        if True: # flag is None or flag=='ex_path_goal_cb':
            self.set_flag('goal_callback')
        # self.subscriber.unregister()

    @smach.cb_interface(input_keys=['path'])
    def ex_path_goal_cb(self, userdata, goal):
        goal.path = userdata.path
        goal.controller = 'controller'

        self.set_flag('ex_path_goal_cb')

    @smach.cb_interface(
        output_keys=['outcome', 'message', 'final_pose', 'dist_to_goal'],
        outcomes=['succeeded', 'failure'])
    def ex_path_result_cb(self, userdata, status, result):
        userdata.message = result.message
        userdata.outcome = result.outcome
        userdata.dist_to_goal = result.dist_to_goal
        userdata.final_pose = result.final_pose

        if result.outcome == ExePathResult.SUCCESS:
            self.outcome = 'succeeded'
        else:
            self.outcome = 'failure'
        # print(self.outcome)

        flag = self.get_flag()
        if flag != 'goal_callback':
            self.set_flag('ex_path_result_cb')
        return self.outcome

def main():
    rospy.init_node('mbf_state_machine')

    mbf_sm = smach.StateMachine(outcomes=['preempted', 'succeeded', 'aborted'])
    mbf_sm.userdata.recovery_flag = False

    with mbf_sm:
        smach.StateMachine.add('WAIT_FOR_GOAL',
                               WaitForGoal(),
                               transitions={'succeeded': 'GET_PATH', 'preempted': 'preempted'})

        smach.StateMachine.add('GET_PATH',
                               smach_ros.SimpleActionState('move_base_flex/get_path',
                                                           GetPathAction,
                                                           goal_cb=get_path_goal_cb,
                                                           result_cb=get_path_result_cb),
                               transitions={'succeeded': 'EXE_PATH',
                                            'failure': 'WAIT_FOR_GOAL'})

        exec_path_custom_state = ExecPath()
        exec_path_action_state = smach_ros.SimpleActionState(
            'move_base_flex/exe_path',
            ExePathAction,
            input_keys=['path'],
            output_keys=['outcome', 'message', 'final_pose', 'dist_to_goal'],
            outcomes=['succeeded', 'failure'],
            goal_cb=functools.partial(ExecPath.ex_path_goal_cb, exec_path_custom_state),
            result_cb=functools.partial(ExecPath.ex_path_result_cb, exec_path_custom_state))

        exec_path_custom_state.set_simple_action_state(exec_path_action_state)
        smach.StateMachine.add('EXE_PATH',
                               exec_path_custom_state,
                               transitions={'succeeded': 'WAIT_FOR_GOAL',
                                            'failure': 'GET_PATH',
                                            'new_goal': 'GET_PATH'})

        smach.StateMachine.add('RECOVERY',
                               smach_ros.SimpleActionState('move_base_flex/recovery',
                                                           RecoveryAction,
                                                           goal_cb=recovery_goal_cb,
                                                           result_cb=recovery_result_cb),
                               transitions={'succeeded': 'GET_PATH',
                                            'failure': 'WAIT_FOR_GOAL'})

    sis = smach_ros.IntrospectionServer('mbf_state_machine_server', mbf_sm, '/SM_ROOT')
    sis.start()
    outcome = mbf_sm.execute()
    rospy.spin()
    sis.stop()


if __name__ == "__main__":
    while not rospy.is_shutdown():
        main()