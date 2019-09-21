import os
import pickle

pickle_dir = "../script/data/"
pickle_files = [os.path.join(pickle_dir, f) for f in os.listdir(pickle_dir) if
                 os.path.isfile(os.path.join(pickle_dir, f)) and f.split(".")[-1] == 'p' and f.split("/")[-1]!="room_door.p"]

dataset_counter = {"close_room":0, "open_room":0, "corridor":0}
for pickle_file in pickle_files:
    dataset_counter_map = {"close_room": 0, "open_room": 0, "corridor": 0}

    data = pickle.load(open(pickle_file, "rb"))
    for item in data:
        try:
            dataset_counter[item[2]] += 1
            dataset_counter_map[item[2]] += 1
        except Exception as e:
            print e, pickle_file
            continue
    print dataset_counter_map, pickle_file
print dataset_counter

