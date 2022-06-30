import os

DataSet = ["deep", "sift", "spacev", "turing"]
DataSize = [1, 10]
# DataSet = ["spacev"]
# DataSize = [100]

Graph = ["plat", "rank"]

def main():
    stage = "search"

    for datasize in DataSize:
        for dataset in DataSet:
            if (dataset == "gist" and datasize != 1):
                continue
            for graph in Graph:
                program = "./main_" + graph
                command = program + " " + stage + " " + dataset + " " + str(datasize)
                # os.system("cd build && make main && " + command)
                os.system("cd build && " + command)

main()