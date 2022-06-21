import os

# DataSet = ["deep", "turing", "gist"]
# DataSize = [1, 10]
DataSet = ["sift"]
DataSize = [1, 10]

Graph = ["hnsw", "plat"]

def main():
    stage = "both"

    for datasize in DataSize:
        for dataset in DataSet:
            if (dataset == "gist" and datasize == 10):
                continue
            for graph in Graph:
                program = "./main_" + graph
                command = program + " " + stage + " " + dataset + " " + str(datasize)
                # os.system("cd build && make main && " + command)
                os.system("cd build && " + command)

main()