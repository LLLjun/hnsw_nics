import os

def run_single():
    efc = 100
    m = 20
    dataname = "deep"
    datasize = 1
    format = ""
    stage = "search"

    if dataname == "sift":
        format = "uint8"
    else:
        format = "float"
    os.system("cd build && make main && cp main main_run_deep")
    command = "cd build && ./main_run_deep " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " 10"
    if efc > m:
        os.system(command)
    os.system("cd build && rm main_run_deep")


def space_explore(mode):
    max_efc = [300]
    max_m = [20]
    datasets = ["deep"]
    datasize = 1
    format = ""
    k = 100
    stage = "search"
    efs_list = [1000]
    # efs_list = [210]

    for dataname in datasets:
        if dataname == "sift":
            format = "uint8"
        else:
            format = "float"

        for efs in efs_list:
            if mode == "train":
                os.system("cd build && make main && cp main main_c3_base_samq")

                for efc in max_efc:
                    for m in max_m:
                        command = "cd build && ./main_c3_base_samq " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " " + str(k) + " " + str(efs)
                        os.system(command)
            
            if mode == "inference":
                os.system("cd build && make main && cp main main_c3_base_real")

                for efc in max_efc:
                    for m in max_m:
                        command = "cd build && ./main_c3_base_real " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " " + str(k) + " " + str(efs)
                        os.system(command)     


# space_explore("train")
space_explore("inference")

# run_single()