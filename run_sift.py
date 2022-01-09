import os

def run_single(efc, m):
    # efc = 250
    # m = 20
    dataname = "sift"
    datasize = 1
    format = ""
    stage = "search"

    if dataname == "sift":
        format = "uint8"
    else:
        format = "float"

    command = "cd build && ./main_run_sift " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " 10"
    if efc > m:
        os.system(command)
    

def run_refine():
    os.system("cd build && make main && cp main main_run_sift")
    run_single(300, 5)
    run_single(250, 5)
    run_single(150, 5)
    run_single(250, 10)
    run_single(200, 10)
    run_single(250, 15)
    run_single(100, 15)
    run_single(150, 15)
    run_single(150, 25)
    run_single(300, 25)
    os.system("cd build && rm main_run_sift")


def space_explore():
    max_efc = range(50, 301, 50)
    max_m = range(5, 26, 5)
    datasets = ["sift"]
    datasize = 1
    format = ""
    stage = "search"

    os.system("cd build && make main && cp main main_run_sift")

    for dataname in datasets:
        if dataname == "sift":
            format = "uint8"
        else:
            format = "float"

        for efc in max_efc:
            for m in max_m:
                command = "cd build && ./main_run_sift " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " 10"
                if efc > m:
                    os.system(command)

    os.system("cd build && rm main_run_sift")


# space_explore()
run_refine()