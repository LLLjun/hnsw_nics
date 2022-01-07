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
    max_efc = 301
    max_m = 26
    datasets = ["deep", "sift", "turing", "gist"]
    format = ""
    stage = "search"

    for dataname in datasets:
        if dataname == "sift":
            format = "uint8"
        else:
            format = "float"

        for efc in range(50, max_efc, 50):
            for m in range(5, max_m, 5):
                command = "make main && ./main " + stage + " " + dataname + " " + format + " 1 " + str(efc) + " " + str(m) + " 10"
                if efc > m:
                    os.system(command)


space_explore()
# run_single()