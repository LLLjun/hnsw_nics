import os

def main():
    stage = "both"
    Datasets = ["spacev"]
    datasize = 50
    Partgraph = [2, 4, 8]

    os.system("cd build && make main")
    for dataset in Datasets:
        for pg in Partgraph:
            command = "./main " + stage + " " + dataset + " " + str(datasize) + " " + str(pg)
            os.system("cd build && " + command)

main()