import os
from plot import handle_data

def run_single(stage):
    efc_list = [300]
    m_list = [20]
    k_list = [1]
    datasets = ["turing"]
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

                    command = "cd build && ./main_c1_sxi " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " " + str(k)
                    if efc >= (2 * m):
                        os.system(command)



def space_explore(stage):
    efc_list = range(100, 301, 100)
    m_list = [20]
    k_list = [1, 10, 100]
    datasets = ["turing"]
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


# space_explore("build")
# space_explore("search")
run_single("build")
run_single("search")