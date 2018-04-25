import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import torch


class WordEncoding:
    def __init__(self):
        self.sos = "[sos]"
        self.eos = "[eos]"
        self.sentences = ["room", "T junction", "Corner"]
        classes = ["room_right", "room_left",
                        "corner_left", "corner_right",
                        "t_junction_right_forward", "t_junction_right_left", "t_junction_left_forward", "t_junction","noting"]

        self.classes = {char: idx for idx, char in enumerate(classes)}
        self.classes_labels = {idx: char for idx, char in enumerate(classes)}

    def len_classes(self):
        return len(self.classes)

    def visualize_map(self, map_data, class_info_list, pose_info_list, base_line_class, base_line_pose):
        print "\n\n"
        base_line_class = base_line_class.cpu().data
        map_data = np.reshape(map_data.cpu().data.numpy(), (map_data.shape[2], map_data.shape[3], 1))
        backtorgb = cv.cvtColor(map_data, cv.COLOR_GRAY2RGB)
        predict = []
        target = []
        for index in range (len(class_info_list[0])):
            if class_info_list[0][index] != 8:
                pose = ((pose_info_list.cpu().data[0][index] - 0.5) * 6.0) * 10.0 + 60.0
                pose = pose.type(torch.IntTensor)
                pose = tuple(pose)
                predict.append((pose, self.get_class_char( class_info_list[0][index])))
                cv.circle(backtorgb, pose , 5, (0,0,255))
            if base_line_class[0][index] != 8:
                pose = ((base_line_pose.cpu().data[0][index] - 0.5) * 6.0) * 10.0 + 60.0
                pose = pose.type(torch.IntTensor)
                pose = tuple(pose)
                target.append((pose, self.get_class_char( base_line_class[0][index])))
                cv.circle(backtorgb, pose , 4, (0,255,0))
        cv.imshow("map", backtorgb)
        cv.waitKey(100)
        print ("predict:")
        print predict
        print ("target")
        print target
        print ("finished")
        # plt.show()

    def get_object_class(self, object):
        if object[1] in self.classes:
            return self.classes[object[1]], object[2]
        else:
            print object
            print ("set_objcet_class class not found skiping")

    def get_class_char(self, class_label):
        if class_label in self.classes_labels:
            return self.classes_labels[class_label]
        else:
            return -1
