import os

def run_single():
    efc = 60
    m = 20
    dataname = "sift"
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
    efc_list = range(60, 81, 20)
    m_list = [20]
    # efc_list = range(50, 301, 50)
    # m_list = range(5, 26, 5)
    datasets = ["sift"]
    datasize = 10
    format = ""

    # os.system("cd build && make main && cp main main_run_sift")

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
    # os.system("cd build && rm main_run_sift")


space_explore("build")
space_explore("search")
# run_single()