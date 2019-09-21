import pickle
import os
import numpy as np
import copy
import matplotlib

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import scipy.stats as stats

from scipy.stats import norm

if __name__ == "__main__":
    lang_dic = {"close_room": 0, "open_room": 1, "corridor": 2}
    all_results_folders = os.listdir("data/map_results/")
    all_results_folders = os.listdir("map_results/")
    models = ["laser", "map", "full"]

    plt.rc('text', usetex=False)
    plt.rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
    matplotlib.rc('xtick', labelsize=60)
    matplotlib.rc('ytick', labelsize=60)
    plt.rcParams["figure.figsize"] = (70, 70)
    f, axarr = plt.subplots(len(models), 2, sharey=True)

    for test_alaki in ("test","train"):
        for explore in ("map_results_unexplored", "map_results_explored"):
            distance_evaluation_all = {x: [] for x in lang_dic}
            distance_evaluation_all = {x: copy.deepcopy(distance_evaluation_all) for x in models}
            distance_evaluation_all = {x: copy.deepcopy(distance_evaluation_all) for x in ["train", "test"]}
            avg_table = {"train":np.zeros((3, 3)), "test":np.zeros((3,3))}
            for folder in all_results_folders:
                folder_dir = "map_results/"+folder+"/" + explore+"/"
                all_pickle = os.listdir(folder_dir)

                all_pickle = [(x, pickle.load(open( os.path.join(folder_dir, x), "rb"))) for x in all_pickle]
                lang_dic = {"close_room":0, "open_room":1, "corridor":2}
                class_evaluation_all = {x:np.zeros(5) for x in lang_dic}
                class_evaluation_all = {x:copy.deepcopy(class_evaluation_all) for x in models}
                class_evaluation_all = {x:copy.deepcopy(class_evaluation_all) for x in ["train", "test"]}
                maps = [
                    "real_1",
                    "real_2",
                    "as1",
                    "fr52",
                    "fr79",
                    "aces_map",
                    "combinenormal"
                ]
                class_evaluation_maps = {x:copy.deepcopy(class_evaluation_all) for x in maps}



                for file in all_pickle:
                    train_test = "train"
                    if "test" in file[0]:
                        train_test = "test"
                    # if "as1" in file[0]:
                    #     train_test = "train"
                    #     if "train" in file[0]:
                    #         train_test = "test"

                    model = "map"
                    if "laser" in file[0]:
                        model = "laser"
                    elif "full" in file[0]:
                        model = "full"
                    # print (file[0])
                    for lang in lang_dic:
                        if (lang in file[1]):
                            file[1][lang]["each_acc_classes"] += [file[1][lang]["no_matched"]]
                            class_evaluation_all[train_test][model][lang] += np.asarray(file[1][lang]["each_acc_classes"])
                            distance_evaluation_all[train_test][model][lang] += (file[1][lang]["distance"])
                            # print (lang, file[1][lang]["each_acc_classes_percent"][lang_dic[lang]])
                            # print (file[1][lang]["each_acc_classes_percent"], file[1][lang]["no_matched"])
                            for map in maps:
                                if map in file[0]:
                                    class_evaluation_maps[map][train_test][model][lang]+= np.asarray(file[1][lang]["each_acc_classes"])

                    # print("")
                # collabel = []
                # for key, value in sorted(lang_dic.iteritems(), key=lambda (k, v): (v, k)):
                #     collabel.append(key)
                #
                # os.system("mkdir -p script/data/plots/" + folder+"/")
                # f, axarr = plt.subplots(len(class_evaluation_maps),2)
                # save_pickle = {}
                # for index_m, map in enumerate(class_evaluation_maps):
                #     for index_t, train_test in enumerate(["train", "test"]):
                #         table = np.zeros((5,3))
                #         for lang in lang_dic:
                #             for y_indext, model in enumerate(models):
                #                 # class_evaluation = [round(float(x) / np.sum(class_evaluation_maps[map][train_test][model][lang]),2) for x in
                #                 #                     class_evaluation_maps[map][train_test][model][lang]]
                #
                #                 class_evaluation = [float(x) for x in
                #                                     class_evaluation_maps[map][train_test][model][lang]]
                #                 table[y_indext, lang_dic[lang]] =  class_evaluation[lang_dic[lang]]
                #                 table[3, lang_dic[lang]] = max(class_evaluation[lang_dic[lang]] + class_evaluation[3], table[3, lang_dic[lang]])
                #                 table[4, lang_dic[lang]] = max(class_evaluation[lang_dic[lang]] + abs(class_evaluation[3] - class_evaluation[4]), table[4, lang_dic[lang]])
                #
                #         axarr[index_m, index_t].table(cellText=table, rowLabels=models+["all"]+["full fp"], colLabels=collabel, loc='center')
                #         axarr[index_m , index_t].set_title(explore+ "_" + map + "_" + train_test, fontsize=5, y=1.24)
                #         save_pickle[explore+ "_" + map + "_" + train_test ] = table
                #         axarr[index_m , index_t].axis('off')


                # f.subplots_adjust(hspace=1.4, wspace=0.5)
                # plt.axis('off')
                # plt.savefig("script/data/plots/"+folder + "/"+ explore+".pdf")
                # pickle.dump(save_pickle, open(("script/data/plots/"+folder + "/"+ explore+".pkl"), "wb"))
                #
                # plt.close()
                #
                # print("script/data/plots/"+folder + "/"+ explore+".pdf")

                for train_test in ("train", "test"):
                    # tables = {"classes":np.zeros((3,3)), "distance_avg": np.zeros((3,3)),"distance_mean": np.zeros((3,3)),
                    #           "distance_max": np.zeros((3,3)),"distance_min": np.zeros((3,3))}
                    tables = {"classes": np.zeros((3, 3))}

                    for lang in lang_dic:
                        for y_indext, model in enumerate (models):
                            class_evaluation =  [float(x)/np.sum(class_evaluation_all[train_test][model][lang][0:4]) for x in class_evaluation_all[train_test][model][lang]]
                            tables["classes"][y_indext, lang_dic[lang]] = class_evaluation[lang_dic[lang]]



                    # for data in tables:
                    #     the_table = plt.table(cellText=tables[data], rowLabels=models, colLabels=collabel, loc='center')
                        # plt.axis('off')
                        # plt.savefig("script/data/plots/"+ folder + "/"+  explore+ data +"_" + train_test + ".pdf")
                        # plt.close()
                        # print ("script/data/plots/"+ folder + "/"+  explore+ data +"_" + train_test + ".pdf")

                    # avg_table[train_test] += tables["classes"]
            # for data in avg_table:
            #     the_table = plt.table(cellText=avg_table[data]/len(all_results_folders), rowLabels=models, colLabels=collabel, loc='center')
            #     plt.axis('off')
            #     plt.savefig("script/data/plots/" + explore + data + ".pdf")
            #     plt.close()
            #     print ("script/data/plots/" + folder + "/" + explore + data + "_" + train_test + ".pdf")
            import matplotlib as mpl





            for t_index, train_test in enumerate(["train", "test"]):

                for y_indext, model in enumerate(models):
                    if train_test != test_alaki:
                        continue
                    t_index = 0 if explore == "map_results_unexplored" else 1

                    distances = []
                    for lang in lang_dic:

                        distance_evaluation_all[train_test][model][lang] = [x for x in
                                                                            distance_evaluation_all[train_test][model][lang]
                                                                            if x != -1]
                        distances += sorted(distance_evaluation_all[train_test][model][lang])

                    n, bins, patches = axarr[y_indext, t_index].hist(distances, 20, normed=1, facecolor='blue', alpha=0.75)
                    # print bins, patches
                    mu, sigma = norm.fit(distances, floc=np.mean(distances))
                    y = mlab.normpdf(bins, mu, sigma)

                    # axarr[y_indext, t_index].plot(bins, y, 'r--', linewidth=2)


                    # fit = stats.norm.pdf(distances, np.mean(distances), np.std(distances))
                    # axarr[y_indext, lang_dic[lang]].hist(distances, normed=True,bins=[x/2. for x in range(0, 20, 1)])
                    # axarr[y_indext, lang_dic[lang]].plot(distances, fit, '-o')
                    #
                    axarr[y_indext, t_index].margins(0.75)

                    axarr[y_indext, t_index].set_title( r' $\mu={},\ \sigma={}$'.format( round(mu, 2), round(sigma, 2)), fontsize=60, y=0.75           )

                    axarr[y_indext, t_index].axis([0, 9, 0.001, 0.4])
                    # if lang_dic[lang] != 0:
                    #     axarr[y_indext, lang_dic[lang]].axis('off')

        # axarr[0, 0].set_xlabel('Train',fontsize=60, y_indext=0)
        # axarr[0, 1].set_xlabel('Test',fontsize=60, y_indext=-10)

        axarr[0, 0].set_ylabel('Laser model\n probability\n',fontsize=60)

        axarr[2, 0].set_xlabel('\n distance (m)',fontsize=60, x=1)
        axarr[1, 0].set_ylabel('Map model\n probability\n', fontsize=60)

        axarr[2, 0].set_ylabel('Combined model\n probability\n', fontsize=60)

        plt.suptitle('\n\n\n\n\n\nUnknown                                                                                           Explored', fontsize = 60)
        f.subplots_adjust(hspace=.1, wspace=.05)
            # f.ylabel('Probability')

            # plt.ylabel('Probability')

        plt.savefig("script/data/plots/" + test_alaki + "_hist" + ".pdf", papertype="a5", dpi=1000)
        plt.close()








    #
    #
    # collabel = []
    # for key, value in sorted(lang_dic.iteritems(), key=lambda (k, v): (v, k)):
    #     collabel.append(key)
    #
    # all_results_folders = os.listdir("script/data/plots/")
    # dic = pickle.load(open( os.path.join("script/data/plots/"+all_results_folders[0]+"/" + "map_results_unexplored")+".pkl", "rb"))
    # dic2 = pickle.load(open( os.path.join("script/data/plots/"+all_results_folders[0]+"/" + "map_results_explored")+".pkl", "rb"))
    # dic.update(dic2)
    # dic = {x:np.zeros(dic[x].shape) for x in dic}
    # for folder in all_results_folders:
    #     for explore in ("map_results_unexplored", "map_results_explored"):
    #         pic = pickle.load(open( os.path.join("script/data/plots/"+folder, explore)+".pkl", "rb"))
    #         for key in pic:
    #             dic[key] +=  pic[key]
    # for key in dic:
    #     dic[key] = dic[key]/len(all_results_folders)
    #     for i in range(3):
    #         for j in range(3):
    #             if dic[key][4][j] != 0:
    #                 dic[key][i][j] = round(dic[key][i][j] /dic[key][3][j], 2)
    #             else:
    #                 dic[key][i][j] = -1
    # f, axarr = plt.subplots(7, 2)
    # index = 0
    # key_sorted = []
    # for key in sorted(dic):
    #     key_sorted.append(key)
    #
    # for key in key_sorted:
    #     if "unexplored" in key:
    #         a = axarr[index/2, index%2].table(cellText=dic[key], rowLabels=["laser", "map", "full"]+["all"]+["full fp"], colLabels=collabel, loc='center')
    #         a.set_fontsize(4)
    #         axarr[index/2, index%2].set_title(key, fontsize=5, y=1.34)
    #         axarr[index/2, index%2].axis('off')
    #         index += 1
    #
    # f.subplots_adjust(hspace=1.7, wspace=0.5)
    # plt.savefig("script/data/plots/1/unexplored.pdf")
    # plt.close()
    #
    # f, axarr = plt.subplots(7, 2)
    # index = 0
    #
    # for key in key_sorted:
    #     if "unexplored" not in key:
    #         a = axarr[index/2, index%2].table(cellText=dic[key], rowLabels=["laser", "map", "full"]+["all"]+["full fp"], colLabels=collabel, loc='center')
    #         a.set_fontsize(4)
    #
    #         axarr[index/2, index%2].set_title(key, fontsize=5, y=1.34)
    #         axarr[index/2, index%2].axis('off')
    #
    #         index += 1
    #
    # f.subplots_adjust(hspace=1.7, wspace=0.5)
    # plt.savefig("script/data/plots/1/explored.pdf")
    # plt.close()
    #
    #
    #
    #
