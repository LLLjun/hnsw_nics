import os

def run_single():
    efc = 60
    m = 15
    datasize = 10
    dataname = "deep"
    format = ""
    stage = "both"

    if dataname == "sift":
        format = "uint8"
    else:
        format = "float"

    command = "cd build && make main && ./main " + stage + " " + dataname + " " + str(datasize) + " " + format + " 1 " + str(efc) + " " + str(m) + " 10"
    if efc > m:
        os.system(command)


def space_explore():
    max_efc = range(30, 51, 20)
    max_m = [15]
    datasize = 10
    datasets = ["deep"]
    format = ""
    stage = "both"

    os.system("cd build && make main && cp main main_run")

    for dataname in datasets:
        if dataname == "sift":
            format = "uint8"
        else:
            format = "float"

        for efc in max_efc:
            for m in max_m:
                command = "cd build && ./main_run " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " 10"
                if efc > m:
                    os.system(command)

    os.system("cd build && rm main_run")

space_explore()
# run_single()