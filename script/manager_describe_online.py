import subprocess
import time
import os

python_running = os.popen(" pgrep -f python").readlines()
python_running = [x.replace("\n", "") for x in python_running]

all_ros_required_nodes = [
                            "/cost_map_node",
                            "/joy_node",
                            "/link1_broadcaster",
                            "/map_server",
                            "/mbf_state_machine",
                            "/move_base_flex",
                            "/rosout",
                            "/rviz",
                            "/slam_gmapping",
                            "/stageros_node",
                            "/teleop_node"
                        ]
def start_process(map_index, model, file_index):
    process_str = 'python script/describe_online.py --map_index ' + str(map_index) + " --file_index " +str(file_index)+' --model ' + model
    print process_str
    p = subprocess.Popen(process_str, stdout=None, shell=True)
    time.sleep(90)

def kill_all():
    nodes = os.popen("rosnode list").readlines()
    for i in range(len(nodes)):
        nodes[i] = nodes[i].replace("\n", "")

    for node in nodes:
        os.system("rosnode kill " + node)
    kill_after_error()

def kill_after_error():
    os.system(" pgrep -f cost_map | xargs kill -9")
    os.system(" pgrep -f stage | xargs kill -9")
    os.system(" pgrep -f rviz | xargs kill -9")
    os.system(" pgrep -f roscore | xargs kill -9")
    os.system(" pgrep -f rosmaster | xargs kill -9")
    python_terminate = os.popen(" pgrep -f python").readlines()
    python_terminate = [x.replace("\n", "") for x in python_terminate]
    print python_terminate
    for pid in python_terminate:
        if pid not in python_running:
            print ("kill -9 " + pid)
            try:
                os.system("kill -9 " + pid)
            except Exception as e:
                print e
                continue

if __name__=="__main__":
    kill_after_error()
    f = open("script/data/map.info", "r")
    start_file_index = 3
    end_file_index = 21
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
    for file_index in range (start_file_index, end_file_index):
        save_path = os.path.join("map_results/", str(file_index)+"/map_results_explored/")
        for map_index in range(len(maps)):
            for model in models:
                # model = "full"
                start_process(map_index, model, file_index)
                time_out = 0

                ls = os.popen("ls " + save_path + " -ntr  | wc -l").readlines()
                ls = int(ls[0].replace("\n", ""))

                while True:
                    try:
                        status = {x:0 for x in all_ros_required_nodes}
                        nodes = os.popen("rosnode list").readlines()
                        for i in range(len(nodes)):
                            nodes[i] = nodes[i].replace("\n", "")
                        number_died_nodes = 0
                        ls_new = os.popen("ls " +save_path + " -ntr  | wc -l").readlines()
                        ls_new = int(ls_new[0].replace("\n", ""))
                        if ls_new > ls:
                            break
                        for node in nodes:
                            status[node] = 1
                        for node in status:
                            if status[node] == 0:
                                number_died_nodes += 1
                        if number_died_nodes > 0:
                            time_out += 1
                            print ("some node died", number_died_nodes, "waiting", time_out, "out of", 60)
                        if time_out > 60:
                            print ("reset the current map by calling emergency kill as timeout and nodes was killed", number_died_nodes, status)
                            kill_all()
                            time_out = 0
                            start_process(map_index, model, file_index)
                        time.sleep(1)
                    except Exception as e:
                        print e, "reset the current map by calling emergency kill"
                        kill_after_error()
                        time.sleep(30)
                        start_process(map_index, model, file_index)
                #killall ros nodes
                try:
                    kill_all()
                except Exception as e:
                    print e, "kill all failed calling emergency kill and continue"
                    kill_after_error()
                    time.sleep(30)
                    continue