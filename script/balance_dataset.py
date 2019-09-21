import pickle
import os
import argparse
import shutil

if __name__=='__main__':

    parser = argparse.ArgumentParser(description='balance data')
    parser.add_argument('--dataset_dir', type=str , default="../data/dataset/train/")
    args = parser.parse_args()
    all_file = len(os.listdir(args.dataset_dir))
    counter_intersection = 0
    counter_empty = 0
    classes = {"close_room":[], "open_room":[], "corridor":[]}
    print (len(os.listdir(args.dataset_dir)))
    for file in os.listdir(args.dataset_dir):
        with open(os.path.join(args.dataset_dir, file), 'rb') as f:
            # repeat = False
            p_file = pickle.load(f)
            if not len(p_file["language"]):

                counter_empty+=1
                print "empty", counter_empty
            for lang in p_file["language"]:
                classes[lang[1]].append(lang)
            #     if 'junction' in lang[1]:
            #         repeat = True
            #         break
            # if repeat:
            #     counter_intersection += 1
                # shutil.copy(os.path.join(args.dataset_dir, file), os.path.join(args.dataset_dir, file+"_copy"))
                # print (counter_intersection, "out of ", all_file)
    # print(len(os.listdir(args.dataset_dir)))
    for item in classes:
        print (item,  len(classes[item]))