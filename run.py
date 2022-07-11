import os

DataSet = ["deep", "sift", "spacev", "turing"]
DataSize = [1, 10, 100]
# DataSet = ["deep"]
# DataSize = [1]
ThreadSize = range(2, 9, 1)

Graph = ["hnsw"]

def main():
    stage = "search"

    for datasize in DataSize:
        for dataset in DataSet:
            for ts in ThreadSize:
                if (dataset == "gist" and datasize != 1):
                    continue
                for graph in Graph:
                    program = "./main_" + graph
                    command = program + " " + stage + " " + dataset + " " + str(datasize) + " " + str(ts)
                    # os.system("cd build && make main && " + command)
                    os.system("cd build && " + command)

main()