import os

def run_single():
    efc = 60
    m = 20
    dataname = "gist"
    datasize = 1
    format = ""
    stage = "build"

    if dataname == "sift":
        format = "uint8"
    else:
        format = "float"

    os.system("cd build && make main && cp main main_run_gist")
    command = "cd build && ./main_run_gist " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " 10"
    if efc > m:
        os.system(command)
    os.system("cd build && rm main_run_gist")


def space_explore(stage):
    efc_list = range(120, 361, 60)
    m_list = [60]
    # efc_list = range(50, 301, 50)
    # m_list = range(5, 26, 5)
    datasets = ["gist"]
    datasize = 1
    format = ""

    # os.system("cd build && make main && cp main main_run_gist")

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

    # efc_list = [60]
    for dataname in datasets:
        if dataname == "sift":
            format = "uint8"
        else:
            format = "float"

        for m in m_list:
            for efc in efc_list:
                command = "cd build && ./main_exi " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " 10"
                if efc >= (2 * m):
                    os.system(command)
    # os.system("cd build && rm main_run_gist")


# space_explore("build")
space_explore("search")
# run_single()