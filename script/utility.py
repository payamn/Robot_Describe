import math
from tf import TransformListener
import time
import rospy
mapper_constant_val_x = 10
mapper_constant_val_y = 10
intersection_r = 2.5
degree_delta = 25
LASER_SCAN_WINDOW = 40
tf_listner = TransformListener()

class Utility:

    @staticmethod
    def get_robot_pose():
        global tf_listner

        number_of_try = 5
        while number_of_try > 0:
            try:
                t = tf_listner.getLatestCommonTime("/robot_0/base_link", "/map_server")
                t = rospy.Time(0)
                position, quaternion = tf_listner.lookupTransform("/map_server", "/robot_0/base_link", t)
            except Exception as e:
                print e
                number_of_try -= 1
                time.sleep(0.2)
                continue
            break
        else:
            raise Exception('tf problem in get_robot_pose.')

        return position, quaternion
    @staticmethod
    def distance_vector(a, b):
        return math.sqrt((a[0]-b[0])*(a[0]-b[0]) + (a[1]-b[1])*(a[1]-b[1]))

    @staticmethod
    def val_vector(a):
        return math.sqrt(a[0] * a[0] + a[1] * a[1])

    @staticmethod
    def dot_product(a, b):
        return a[0] * b[0] + a[1] * b[1]

    @staticmethod
    def degree_vector(a, b):
        return math.atan2(a[1], a[0]) - math.atan2(b[1], b[0])

    @staticmethod
    def degree_norm(degree):
        return math.atan2(math.sin(degree), math.cos(degree))

    @staticmethod
    def in_threshold(x, num, threshold):
        if math.fabs(x - num) < threshold:
            return True
        return False

    @staticmethod
    def quaternion_to_euler_angle(w, x, y, z):
        ysqr = y * y

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + ysqr)
        X = math.degrees(math.atan2(t0, t1))

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        Y = math.degrees(math.asin(t2))

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (ysqr + z * z)
        Z = math.degrees(math.atan2(t3, t4))

        return X, Y, Z