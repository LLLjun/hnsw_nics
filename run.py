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
    efc_list = range(60, 101, 40)
    m_list = [20]
    datasets = ["sift"]
    datasize = 10
    k_list = [1, 10, 100]
    format = ""

    for dataname in datasets:
        if dataname == "sift":
            format = "uint8"
        else:
            format = "float"

        for efc in efc_list:
            for m in m_list:
                for k in k_list:
                    command = "cd build && ./main_base " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " " + str(k)
                    if efc > m:
                        os.system(command)

                    command = "cd build && ./main_exi " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " " + str(k)
                    if efc > m:
                        os.system(command)

    # efc_list = range(80, 201, 40)
    # m_list = [40]
    # datasets = ["gist"]
    # datasize = 1
    # format = ""

    # for dataname in datasets:
    #     if dataname == "sift":
    #         format = "uint8"
    #     else:
    #         format = "float"

    #     for efc in efc_list:
    #         for m in m_list:
    #             command = "cd build && ./main_base " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " 10"
    #             if efc > m:
    #                 os.system(command)

    #             command = "cd build && ./main_exi " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " 10"
    #             if efc > m:
    #                 os.system(command)

# space_explore("build")
space_explore("search")
# run_single()