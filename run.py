import os

def main():
    stage = "both"
    Datasets = ["sift", "spacev"]
    datasize = 100
    # Sub_graph = [2, 4, 8]
    Sub_graph = [16]

    os.system("cd build && make main")
    for dataset in Datasets:
        for sg in Sub_graph:
            command = "./main " + stage + " " + dataset + " " + str(datasize) + " " + str(sg)
            os.system("cd build && " + command)

main()