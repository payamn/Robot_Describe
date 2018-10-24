
import pickle
import os
import numpy as np
import copy
import matplotlib.pyplot as plt

if __name__ == "__main__":
    all_results_folders = os.listdir("data/map_results/")
    for folder in all_results_folders:
        all_pickle = os.listdir("data/map_results/"+folder+"/")
        all_pickle = [(x, pickle.load(open( os.path.join("data/map_results/",folder, x), "rb"))) for x in all_pickle]
        lang_dic = {"close_room":0, "open_room":1, "corridor":2}
        models = ["laser", "map", "full"]
        class_evaluation_all = {x:np.zeros(4) for x in lang_dic}
        class_evaluation_all = {x:copy.deepcopy(class_evaluation_all) for x in models}
        class_evaluation_all = {x:copy.deepcopy(class_evaluation_all) for x in ["train", "test"]}
        maps = [
            "real_1",
            "real_2",
            "room_door",
            "as1",
            "fr52",
            "fr79",
            "aces_map",
            "combinenormal"
        ]
        class_evaluation_maps = {x:copy.deepcopy(class_evaluation_all) for x in maps}

        distance_evaluation_all = {x:[] for x in lang_dic}
        distance_evaluation_all = {x:copy.deepcopy(distance_evaluation_all) for x in models}
        distance_evaluation_all = {x:copy.deepcopy(distance_evaluation_all) for x in ["train", "test"]}

        for file in all_pickle:
            train_test = "train"
            if "aces_map" in file[0] or "combinenormal" in file[0]:
                continue
            if "test" in file[0]:
                train_test = "test"

            model = "map"
            if "laser" in file[0]:
                model = "laser"
            elif "full" in file[0]:
                model = "full"
            # print (file[0])
            for lang in lang_dic:
                if (lang in file[1]):
                    class_evaluation_all[train_test][model][lang] += np.asarray(file[1][lang]["each_acc_classes"])
                    distance_evaluation_all[train_test][model][lang] += (file[1][lang]["distance"])
                    # print (lang, file[1][lang]["each_acc_classes_percent"][lang_dic[lang]])
                    # print (file[1][lang]["each_acc_classes_percent"], file[1][lang]["no_matched"])
                    for map in maps:
                        if map in file[0]:
                            class_evaluation_maps[map][train_test][model][lang]+= np.asarray(file[1][lang]["each_acc_classes"])

            # print("")
        collabel = []
        for key, value in sorted(lang_dic.iteritems(), key=lambda (k, v): (v, k)):
            collabel.append(key)

        # for map in class_evaluation_maps:
        #     for train_test in ("train", "test"):
        #         table = np.zeros((3,3))
        #         for lang in lang_dic:
        #             for y_indext, model in enumerate(models):
        #                 class_evaluation = [round(float(x) / np.sum(class_evaluation_maps[map][train_test][model][lang]),2) for x in
        #                                     class_evaluation_maps[map][train_test][model][lang]]
        #                 table[y_indext, lang_dic[lang]] =  class_evaluation[lang_dic[lang]]
        #         the_table = plt.table(cellText=table, rowLabels=models, colLabels=collabel, loc='center')
        #         plt.axis('off')
        #         plt.savefig("script/data/plots/" + folder + map + "_" + train_test + ".pdf")
        #         plt.close()

        #         print("script/data/plots/" + folder + map + "_" + train_test + ".pdf")

        for train_test in ("train", "test"):
            # tables = {"classes":np.zeros((3,3)), "distance_avg": np.zeros((3,3)),"distance_mean": np.zeros((3,3)),
            #           "distance_max": np.zeros((3,3)),"distance_min": np.zeros((3,3))}
            tables = {"classes": np.zeros((3, 3))}

            for lang in lang_dic:
                for y_indext, model in enumerate (models):
                    distance_evaluation_all[train_test][model][lang] = [x for x in distance_evaluation_all[train_test][model][lang] if x != -1]
                    # print lang, model, train_test
                    # print class_evaluation_all[train_test][model][lang], np.mean(distance_evaluation_all[train_test][model][lang]), np.min(class_evaluation_all[train_test][model][lang]), np.max(class_evaluation_all[train_test][model][lang])
                    class_evaluation =  [round(float(x)/np.sum(class_evaluation_all[train_test][model][lang]),2) for x in class_evaluation_all[train_test][model][lang]]
                    tables["classes"][y_indext, lang_dic[lang]] = class_evaluation[lang_dic[lang]]
                    # tables["distance_min"][y_indext, lang_dic[lang]] = np.min(distance_evaluation_all[train_test][model][lang])
                    # tables["distance_avg"][y_indext, lang_dic[lang]] = np.average(distance_evaluation_all[train_test][model][lang])
                    # tables["distance_max"][y_indext, lang_dic[lang]] = np.max(distance_evaluation_all[train_test][model][lang])
                    # tables["distance_mean"][y_indext, lang_dic[lang]] = np.median(distance_evaluation_all[train_test][model][lang])
                    # print class_evaluation[lang_dic[lang]]
                    # print ("")
            for data in tables:
                the_table = plt.table(cellText=tables[data], rowLabels=models, colLabels=collabel, loc='center')
                plt.axis('off')
                plt.savefig("script/data/plots/"+folder + data +"_" + train_test + ".pdf")
                plt.close()
                print ("script/data/plots/"+folder + data +"_" + train_test + ".pdf")
    #
    # fig, axs = plt.subplots(2, 2)
    # clust_data = np.random.random((3, 3))
    # # axs[0,0].axis('tight')
    # axs[0,0].axis('off')
    # the_table = plt.table(cellText=clust_data, rowLabels= models, colLabels=collabel, loc='center')
    # plt.axis('off')
    # plt.savefig("script/data/plots/test.png")
    # axs[1,1].plot(clust_data[:, 0], clust_data[:, 1])
    # plt.show()