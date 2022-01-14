import os

def run_single():
    efc = 40
    m = 20
    datasize = 10
    dataname = "deep"
    format = ""
    stage = "build"

    if dataname == "sift":
        format = "uint8"
    else:
        format = "float"

    os.system("cd build && make main && cp main main_run")
    command = "cd build && ./main_run " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " 10"
    if efc > m:
        os.system(command)
    os.system("cd build && rm main_run")


def space_explore(stage):
    efc_list = range(40, 101, 20)
    m_list = [20]
    datasets = ["deep"]
    datasize = 1
    format = ""
    # stage = "both"

    os.system("cd build && make main && cp main main_run")

    for dataname in datasets:
        if dataname == "sift":
            format = "uint8"
        else:
            format = "float"

        for efc in efc_list:
            for m in m_list:
                command = "cd build && ./main_run " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " 10"
                if efc > m:
                    os.system(command)

    os.system("cd build && rm main_run")

space_explore("build")
space_explore("search")
# run_single()