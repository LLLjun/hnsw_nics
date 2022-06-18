import os

DataSet = ["deep", "turing", "gist"]
DataSize = [1, 10]

def main():
    stage = "both"

    for datasize in DataSize:
        for dataset in DataSet:
            if (dataset == "gist" and datasize == 10):
                continue

            command = "./main " + stage + " " + dataset + " " + str(datasize)
            os.system("cd build && make main && " + command)

main()