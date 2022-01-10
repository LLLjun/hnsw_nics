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


def space_explore(stage):
    max_efc = range(400, 701, 100)
    max_m = range(30, 51, 5)
    datasets = ["deep", "sift", "gist", "turing"]
    datasize = 1
    format = ""

    os.system("cd build && make main && cp main main_run")

    for dataname in datasets:
        if dataname == "sift":
            format = "uint8"
        else:
            format = "float"

        for m in max_m:
            for efc in max_efc:
                command = "cd build && ./main_run " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " 10"
                if efc > m:
                    os.system(command)

    os.system("cd build && rm main_run")


space_explore("build")
space_explore("search")
# run_single()