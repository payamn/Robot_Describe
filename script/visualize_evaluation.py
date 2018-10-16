
import pickle
import os
import numpy as np
import copy
import matplotlib.pyplot as plt

if __name__ == "__main__":
    all_pickle = os.listdir("map_results/")
    all_pickle = [(x, pickle.load(open( os.path.join("map_results/",x), "rb"))) for x in all_pickle]
    lang_dic = {"close_room":0, "open_room":1, "corridor":2}
    models = ["laser", "map", "full"]
    class_evaluation_all = {x:np.zeros(4) for x in lang_dic}
    class_evaluation_all = {x:copy.deepcopy(class_evaluation_all) for x in models}
    class_evaluation_all = {x:copy.deepcopy(class_evaluation_all) for x in ["train", "test"]}
    distance_evaluation_all = {x:[] for x in lang_dic}
    distance_evaluation_all = {x:copy.deepcopy(distance_evaluation_all) for x in models}
    distance_evaluation_all = {x:copy.deepcopy(distance_evaluation_all) for x in ["train", "test"]}

    for file in all_pickle:
        train_test = "train"
        if "test" in file[0]:
            train_test = "test"

        model = "map"
        if "laser" in file[0]:
            model = "laser"
        elif "full" in file[0]:
            model = "full"
        print (file[0])
        for lang in lang_dic:
            if (lang in file[1]):
                class_evaluation_all[train_test][model][lang] += np.asarray(file[1][lang]["each_acc_classes"])
                distance_evaluation_all[train_test][model][lang] += (file[1][lang]["distance"])
                print (lang, file[1][lang]["each_acc_classes_percent"][lang_dic[lang]])
                print (file[1][lang]["each_acc_classes_percent"], file[1][lang]["no_matched"])

        print("")
    table_data = {"train":np.zeros((3, 3)), "test": np.zeros((3,3))}

    for train_test in ("train", "test"):
        for lang in lang_dic:
            for model in models:
                distance_evaluation_all[train_test][model][lang] = [x for x in distance_evaluation_all[train_test][model][lang] if x != -1]
                print lang, model, train_test
                print class_evaluation_all[train_test][model][lang], np.mean(distance_evaluation_all[train_test][model][lang]), np.min(class_evaluation_all[train_test][model][lang]), np.max(class_evaluation_all[train_test][model][lang])
                class_evaluation =  [float(x)/np.sum(class_evaluation_all[train_test][model][lang]) for x in class_evaluation_all[train_test][model][lang]]
                print class_evaluation[lang_dic[lang]]
                print ("")


    fig, axs = plt.subplots(2, 2)
    clust_data = np.random.random((3, 3))
    collabel = [x for x in lang_dic]
    # axs[0,0].axis('tight')
    axs[0,0].axis('off')
    the_table = axs[0,0].table(cellText=clust_data, rowLabels= models, colLabels=collabel, loc='center')

    axs[1,1].plot(clust_data[:, 0], clust_data[:, 1])
    plt.show()