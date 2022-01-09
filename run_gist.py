import os

def run_single(efc, m):
    # efc = 250
    # m = 20
    dataname = "gist"
    datasize = 1
    format = ""
    stage = "search"

    if dataname == "sift":
        format = "uint8"
    else:
        format = "float"

    command = "cd build && ./main_run_gist " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " 10"
    if efc > m:
        os.system(command)
    

def run_refine():
    os.system("cd build && make main && cp main main_run_gist")
    run_single(250, 20)
    # run_single(250, 25)
    # run_single(200, 25)
    # run_single(150, 25)
    # run_single(100, 25)
    # run_single(50, 25)
    os.system("cd build && rm main_run_gist")

def space_explore():
    max_efc = range(50, 301, 50)
    max_m = range(5, 26, 5)
    datasets = ["gist"]
    datasize = 1
    format = ""
    stage = "search"

    os.system("cd build && make main && cp main main_run_gist")

    for dataname in datasets:
        if dataname == "sift":
            format = "uint8"
        else:
            format = "float"

        for efc in max_efc:
            for m in max_m:
                command = "cd build && ./main_run_gist " + stage + " " + dataname + " " + format + " " + str(datasize) + " " + str(efc) + " " + str(m) + " 10"
                if efc > m:
                    os.system(command)

    os.system("cd build && rm main_run_gist")


space_explore()
# run_refine()