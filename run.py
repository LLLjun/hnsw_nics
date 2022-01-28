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
    efc_list = [300]
    m_list = [20]
    datasets = ["turing"]
    datasize = 1
    k_list = [1, 10]
    format = ""

    for dataname in datasets:
        if dataname == "sift":
            format = "uint8"
        else:
            format = "float"

        if dataname == "gist":
            m_list = [30]

        for efc in efc_list:
            for m in m_list:
                for k in k_list:
                    command = "cd build && ./main_c1_base " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " " + str(k)
                    if efc > m:
                        os.system(command)

                    command = "cd build && ./main_c1_sxi " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " " + str(k)
                    if efc > m:
                        os.system(command)

space_explore("build")
space_explore("search")
# run_single()