import os

def run_single():
    efc = 100
    m = 25
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
    max_efc = [300]
    max_m = [20]
    datasets = ["deep", "sift", "turing"]
    datasize = 1
    format = ""

    for dataname in datasets:
        if dataname == "sift":
            format = "uint8"
        else:
            format = "float"

        for efc in max_efc:
            for m in max_m:
                command = "cd build && ./main_c3_base_samq " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " 10"
                os.system(command)
                command = "cd build && ./main_c3_base_real " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " 10"
                os.system(command)


space_explore("build")
space_explore("search")
# run_single()