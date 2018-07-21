import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch


class WordEncoding:
    def __init__(self):
        self.sos = "[sos]"
        self.eos = "[eos]"
        self.sentences = ["close_room", "open_room", "4_junction" ,"t_junction", "corner"]
        classes = ["close_room", "open_room",
                        "corner_left", "corner_right",
                        "t_junction_right_forward", "t_junction_right_left", "t_junction_left_forward", "t_junction", "4_junction"]

        self.classes = {char: idx for idx, char in enumerate(classes)}
        self.classes_labels = {idx: char for idx, char in enumerate(classes)}
        # self.parent_class_dic = {idx: prt_idx for idx, lable in enumerate(classes) for prt_idx, prt_lable in enumerate(self.sentences) if prt_lable in lable}
    def len_classes(self):
        return len(self.classes)

    # def get_parent_class(self, idx):
    #     return self.parent_class_dic[idx]

    def visualize_map(self, map_data, laser_map,predict_classes, predict_poses, predict_objectness, target_classes, target_poses, target_objectness):
        print "\n\n"
        for batch in range(target_classes.shape[0]):
            predict = []
            target = []
            # map_data = np.reshape(map_data[batch].cpu().data.numpy(),(map_data.shape[1], map_data.shape[2], 1))
            backtorgb = cv.cvtColor(map_data[batch].cpu().data.numpy(), cv.COLOR_GRAY2RGB)
            backtorgb_laser = cv.cvtColor(laser_map[batch].cpu().data.numpy(), cv.COLOR_GRAY2RGB)

            for x in range (target_classes.shape[1]):
                for y in range (target_classes.shape[2]):
                    for anchor in range(target_classes.shape[3]):
                        if (target_objectness[batch][x][y][anchor].item()>= 0.3):
                            pose = ((target_poses[batch][x][y][anchor].cpu().numpy()))
                            pose = (pose + np.asarray([x, y])) * ( float(backtorgb.shape[1]) / target_classes.shape[1])
                            pose = pose.astype(int)
                            pose = tuple(pose)
                            target.append((pose, self.get_class_char(target_classes[batch][x][y][anchor].item())))
                            cv.circle(backtorgb, pose, 5, (0, 0, 255))
                            cv.circle(backtorgb_laser, pose, 5, (0, 0, 255))

                        if (predict_objectness[batch][x][y][anchor].item()>= 0.3):
                            pose = ((predict_poses[batch][x][y][anchor].cpu().detach().numpy()))
                            pose = (pose + np.asarray([x, y])) * ( float(backtorgb.shape[1]) / predict_classes.shape[1])
                            pose = pose.astype(int)
                            pose = tuple(pose)
                            predict.append((pose, self.get_class_char(predict_classes[batch][x][y][anchor].item())))
                            cv.circle(backtorgb, pose, 4, (255, 0, 100))
                            cv.circle(backtorgb_laser, pose, 4, (255, 0, 100))

            cv.imshow("map", backtorgb)
            print ("predict:")
            print predict
            print ("target")
            print target
            cv.imshow("laser map", backtorgb_laser)
            cv.waitKey()

            plt.show()

    def get_object_class(self, object):
        if object[1] in self.classes:
            prefer_anchor = 1
            if "room" in object[1]:
                prefer_anchor = 0
            return self.classes[object[1]], object[2], prefer_anchor
        else:
            print object, 0
            print ("set_objcet_class class not found skiping")

    def get_class_char(self, class_label):
        if class_label in self.classes_labels:
            return self.classes_labels[class_label]
        else:
            return -1

def laser_to_map(laser_array, fov, dest_size, max_range_laser):

    fov = float(fov)
    degree_steps = fov/len(laser_array)
    map = np.zeros((dest_size, dest_size, 1))
    to_map_coordinates = float(dest_size)/8.0
    lasers = [((len(laser_array)/2-index)*degree_steps, x) for index, x in enumerate(laser_array)]

    for laser in lasers:
        x = laser[1] * np.cos(laser[0]/180*np.pi) * max_range_laser
        y = laser[1] * np.sin(laser[0]/180*np.pi) * max_range_laser

        x = x * to_map_coordinates
        y = (-y + 8.0/2.)* to_map_coordinates

        if x >= dest_size or y >= dest_size or x<0 or y<0:
            continue
        cv.circle(map, (int(x), int(y)), 3, 255, -1);

        map[int(y), int(x),0] = 255
    # backtorgb = cv.cvtColor(map, cv.COLOR_GRAY2RGB)


    return map