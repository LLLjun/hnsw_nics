import os

def main():
    stage = "both"
    Datasets = ["sift", "spacev"]
    datasize = 50
    # Sub_graph = [2, 4, 8]
    Sub_graph = [1]

    os.system("cd build && make main")
    for dataset in Datasets:
        for sg in Sub_graph:
            command = "./main " + stage + " " + dataset + " " + str(datasize) + " " + str(sg)
            os.system("cd build && " + command)

main()