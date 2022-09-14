import os

def main():
    stage = "both"
    Datasets = ["sift", "spacev"]
    datasize = 100
    Partgraph = [4, 8]

    for dataset in Datasets:
        for pg in Partgraph:
            command = "./main " + stage + " " + dataset + " " + str(datasize) + " " + str(pg)
            os.system("cd build && make main && " + command)

main()