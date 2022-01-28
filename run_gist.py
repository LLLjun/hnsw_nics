import os
from plot import handle_data

def run_single():
    efc = 60
    m = 20
    dataname = "deep"
    datasize = 1
    format = ""
    stage = "build"

    if dataname == "sift":
        format = "uint8"
    else:
        format = "float"

    os.system("cd build && make main && cp main main_run_sift")
    command = "cd build && ./main_run_sift " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " 10"
    if efc > m:
        os.system(command)
    os.system("cd build && rm main_run_sift")


def space_explore(stage):
    efc_list = [300, 500]
    m_list = [35]
    k_list = [1, 10]
    datasets = ["gist"]
    datasize = 1
    format = ""

    for dataname in datasets:
        for k in k_list:
            if dataname == "sift":
                format = "uint8"
            else:
                format = "float"

            for m in m_list:
                for efc in efc_list:

                    command = "cd build && ./main_c1_base " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " " + str(k)
                    if efc >= (2 * m):
                        os.system(command)
                    
                    command = "cd build && ./main_c1_sxi " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " " + str(k)
                    if efc >= (2 * m):
                        os.system(command)

            if (stage == "search"):
                handle_data(dataname, datasize, efc_list, m_list, k)


space_explore("build")
space_explore("search")
# run_single()
