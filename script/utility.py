import math
from tf import TransformListener
import time
import rospy
import cv2
import numpy as np
import sys
import torch
import os
import shutil

mapper_constant_val_x = 10
mapper_constant_val_y = 10
intersection_r = 2.5
degree_delta = 25
LASER_SCAN_WINDOW = 40
tf_listner = TransformListener()

class CheckPointSaver:
    def __init__(self, metrics, best_metrics=None, model_dir='checkpoints'):
        self.best_metric_values = best_metrics
        self.metric_names = metrics
        self.model_dir = model_dir

        if best_metrics is not None:
            if len(metrics) != len(best_metrics):
                sys.exit('Mismatch len of metric names and values')

    def save_checkpoint(self, state, filename='checkpoint.pth.tar'):
        torch.save(state, os.path.join(self.model_dir, filename))
        for metric_name in (self.metric_names):
            if metric_name in state:
                best_metric_value = self.best_metric_values[metric_name]
                is_best = False
                if best_metric_value is None:
                    is_best = True
                elif 'loss' in metric_name:
                    is_best = state[metric_name] < best_metric_value
                elif 'acc' in metric_name:
                    is_best = state[metric_name] > best_metric_value

                if is_best:
                    # print ("saving", metric_name)
                    shutil.copyfile(os.path.join(self.model_dir, filename), os.path.join(self.model_dir, 'model_best_' + metric_name + '.pth.tar'))
                    self.best_metric_values[metric_name] = state[metric_name]



class Utility:

    @staticmethod
    def normalize(data):
        data -= data.min()
        data /= (data.max() - data.min())
        data *= 255
        data = data.astype(np.uint8)
        return data

    @staticmethod
    def sub_image(image, resolution, center, theta, width, height, only_forward=False, dilate=True, transform=None, resize=None):
        '''
        Rotates OpenCV image around center with angle theta (in deg)
        then crops the image according to width and height.
        '''
        if not theta:
            theta = 0
        width = int(np.ceil(width / resolution))
        height = int(np.ceil(height / resolution))
        center = (center[0] / resolution + height, center[1] / resolution + width)

        # Uncomment for theta in radians
        # theta *= 180/np.pi
        if resize:
            image_r = cv2.resize(image, (0, 0), fx=resize, fy=resize)
            if resize > 1:
                start_height = int(np.ceil((resize - 1) / 2.0 * image.shape[0]))
                start_width = 0
                # start_width = int(np.ceil((resize - 1) / 2.0 * image.shape[1]))
                image = image_r[start_height:image.shape[0] + start_height, start_width:image.shape[1] + start_width]
            elif resize < 1:
                image = np.zeros(image.shape, np.uint8)
                start_height = int(np.ceil((1 - resize) / 2.0 * image.shape[0]))
                start_width = 0
                # start_width = int(np.ceil((1 - resize) / 2.0 * image.shape[1]))
                image[start_height: image_r.shape[0] + start_height, start_width: image_r.shape[1] + start_width] = image_r


        image = cv2.copyMakeBorder(image, top=height, bottom=height, left=width, right=width,
                                   borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

        shape = image.shape[:2]

        # cv2.namedWindow('map1', cv2.WINDOW_NORMAL)
        # cv2.imshow("map1", image)
        # cv2.waitKey(1)
        matrix = cv2.getRotationMatrix2D(center=center, angle=theta, scale=1)
        image = cv2.warpAffine(image, matrix, (shape[1], shape[0]))

        if transform:
            M = np.float32([[1, 0, transform[0]*width], [0, 1, transform[1]*height]])
            image = cv2.warpAffine(image, M, (shape[1], shape[0]))

        # cv2.imshow("map1", image)
        # cv2.waitKey(1)

        x = int(np.ceil(center[0] - width / 2)) if not only_forward else int(np.ceil(center[0]))
        y = int(np.ceil(center[1] - height / 2))

        image = image[y:y + height, x:x + width]
        if dilate:
            kernel = np.ones((10, 10), np.uint8)
            d_im = cv2.dilate(image, kernel, iterations=1)
            image = cv2.erode(d_im, kernel, iterations=1)
        else: # for real map we will reduce the thickness of obstacle

            # kernel = np.ones((10, 10), np.uint8)
            # d_im = cv2.dilate(image, kernel, iterations=1)
            # image = cv2.erode(image, kernel, iterations=1)
            #
            kernel = np.ones((2,2), np.uint8)
            image = cv2.erode(image, kernel, iterations=1)

        return image

    @staticmethod
    def rotate_point(origin, point, angle):
        """
        Rotate a point counterclockwise by a given angle around a given origin.

        The angle should be given in radians.
        """
        ox, oy = origin
        px, py = point

        qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
        qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
        return qx, qy

    @staticmethod
    def get_robot_pose(map_topic):
        global tf_listner

        number_of_try = 5
        while number_of_try > 0:
            try:
                # t = tf_listner.getLatestCommonTime("/base_laser", map_topic)
                t = rospy.Time(0)
                position, quaternion = tf_listner.lookupTransform(map_topic, "/base_laser_link", t)
            except Exception as e:
                print (e)
                number_of_try -= 1
                time.sleep(0.2)
                continue
            break
        else:
            raise Exception('tf problem in get_robot_pose.', map_topic)

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
    def quaternion_to_euler_angle(x, y, z, w):
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

    @staticmethod
    def toQuaternion(pitch, roll, yaw):
        cy = math.cos(yaw * 0.5);
        sy = math.sin(yaw * 0.5);
        cr = math.cos(roll * 0.5);
        sr = math.sin(roll * 0.5);
        cp = math.cos(pitch * 0.5);
        sp = math.sin(pitch * 0.5);
        w = cy * cr * cp + sy * sr * sp;
        x = cy * sr * cp - sy * cr * sp;
        y = cy * cr * sp + sy * sr * cp;
        z = sy * cr * cp - cy * sr * sp;
        return w, x, y, z
