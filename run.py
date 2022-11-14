import os

def main():
    stage = "search"
    Datasets = ["sift", "spacev"]
    Datasize = [10, 100]
    Batchsize = [1, 10]

    os.system("cd build && make main")
    for dataset in Datasets:
        for datasize in Datasize:
            for batchsize in Batchsize:
                command = "./main " + stage + " " + dataset + " " + str(datasize) + " " + str(batchsize)
                os.system("cd build && " + command)

main()