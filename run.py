import os

def main():
    Datasets = ["sift", "spacev"]
    Datasize = [50]
    trans = "no_trans"

    os.system("cd build && make main")
    for datasize in Datasize:
        for dataset in Datasets:
            command = "./main " + dataset + " " + str(datasize) + " " + str(trans)
            os.system("cd build && " + command)

main()