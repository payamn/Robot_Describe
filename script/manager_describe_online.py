import subprocess
import time
import os

def start_process(map_index, model):
    process_str = 'python script/describe_online.py --map_index ' + str(map_index) + ' --model ' + model
    print process_str
    p = subprocess.Popen(process_str, stdout=None, shell=True)
    time.sleep(90)

def kill_all():
    nodes = os.popen("rosnode list").readlines()
    for i in range(len(nodes)):
        nodes[i] = nodes[i].replace("\n", "")

    for node in nodes:
        os.system("rosnode kill " + node)

if __name__=="__main__":

    f = open("script/data/map.info", "r")
    maps = []
    line = f.readline()
    while (line):
        maps.append(line.split())
        line = f.readline()

    maps = [(x[0], float(x[1]), float(x[2]), x[3]) for x in maps]

    models = {"full":"checkpoints_final_my_resnet/model_best_epoch_accuracy_classes.pth.tar",
                   "laser":"checkpoints_laser_final/model_best_epoch_accuracy_classes.pth.tar",
                   "map":"checkpoints_map_final/model_best_epoch_accuracy_classes.pth.tar"}
    global MAP_NAME, OFFSET_MAP, MODE
    for map_index in range(len(maps)):
        for model in models:
            start_process(map_index, model)


            while True:
                nodes = os.popen("rosnode list").readlines()
                # is stage allive
                stage = False
                for node in nodes:
                    if "stage" in node:
                        stage = True
                if not stage and len(nodes) > 8:
                    kill_all()
                    start_process()
                if len(nodes) < 3:
                    break
                time.sleep(5)
            #killall ros nodes
            kill_all()
