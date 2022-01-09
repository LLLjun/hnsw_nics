import os

def run_single():
    efc = 60
    m = 30
    dataname = "deep"
    format = ""
    stage = "both"

    if dataname == "sift":
        format = "uint8"
    else:
        format = "float"

    command = "make main && ./main " + stage + " " + dataname + " " + format + " 10 " + str(efc) + " " + str(m) + " 10"
    if efc > m:
        os.system(command)


def space_explore():
    max_efc = range(50, 301, 50)
    max_m = range(5, 26, 5)
    datasets = ["deep", "sift", "gist"]
    datasize = 1
    format = ""
    stage = "search"

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