import os
from plot import handle_data

def run_single(stage):
    efc_list = [300]
    m_list = [30]
    datasets = ["gist"]
    datasize = 1
    format = ""
    k_list = [1]

    # os.system("cd build && make main && cp main main_run_sift")

    for dataname in datasets:
        if dataname == "sift":
            format = "uint8"
        else:
            format = "float"

        for k in k_list:
            for m in m_list:
                for efc in efc_list:

                    command = "cd build && ./main_c2_base " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " " + str(k)
                    if efc >= (2 * m):
                        os.system(command)
                    
                    command = "cd build && ./main_c2_rldt " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " " + str(k)
                    if efc >= (2 * m):
                        os.system(command)

            if (stage == "search"):
                handle_data(dataname, efc_list, m_list, k)


def space_explore(stage):
    efc_list = range(300, 301, 100)
    m_list = range(30, 31, 10)
    datasets = ["gist"]
    datasize = 1
    format = ""
    k_list = [1, 10]
    dms_list = [1, 3, 7, 15, 31]
    ncf_list = [1, 3, 5, 7, 9]
    # dms_list = [1]
    # ncf_list = [3]

    # os.system("cd build && make main && cp main main_run_sift")

    for dataname in datasets:
        if dataname == "sift":
            format = "uint8"
        else:
            format = "float"

        for k in k_list:
            for m in m_list:
                for efc in efc_list:

                    command = "cd build && ./main_c2_base " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " " + str(k) + " 0 0"
                    if efc >= (2 * m):
                        os.system(command)
                    
                    for dms in dms_list:
                        for ncf in ncf_list:
                            command = "cd build && ./main_c2_rldt " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " " + str(k) + " " + str(dms) + " " + str(ncf)
                            if efc >= (2 * m):
                                os.system(command)

            if (stage == "search"):
                handle_data(dataname, efc_list, m_list, k, dms_list, ncf_list)


space_explore("build")
space_explore("search")
# run_single()
# run_single("build")
# run_single("search")