import os

def main():
    Datasets = ["sift", "spacev"]
    datasize = 1000
    trans = "no_trans"

    for dataset in Datasets:
        command = "./main " + dataset + " " + str(datasize) + " " + str(trans)
        os.system("cd build && make main && " + command)

main()