import os

def run_single():
    efc = 60
    m = 20
    dataname = "turing"
    datasize = 1
    format = ""
    stage = "build"

    if dataname == "sift":
        format = "uint8"
    else:
        format = "float"

    os.system("cd build && make main && cp main main_run_turing")
    command = "cd build && ./main_run_turing " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " 10"
    if efc > m:
        os.system(command)
    os.system("cd build && rm main_run_turing")


def space_explore(stage):
    efc_list = range(40, 101, 20)
    m_list = [20]
    # efc_list = range(50, 301, 50)
    # m_list = range(5, 26, 5)
    datasets = ["turing"]
    datasize = 1
    format = ""

    # os.system("cd build && make main && cp main main_run_turing")

    for dataname in datasets:
        if dataname == "sift":
            format = "uint8"
        else:
            format = "float"

        for m in m_list:
            for efc in efc_list:
                command = "cd build && ./main_base " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " 10"
                if efc >= (2 * m):
                    os.system(command)

                command = "cd build && ./main_exi " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " 10"
                if efc >= (2 * m):
                    os.system(command)
    # os.system("cd build && rm main_run_turing")


# space_explore("build")
space_explore("search")
# run_single()