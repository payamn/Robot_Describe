
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
