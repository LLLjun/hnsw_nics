import os


def testRank():
    stage = "search"

    DataSet = ["deep", "sift", "spacev"]
    DataSize = [1, 10, 100]
    threadsize = 1

    Graph = ["rank_bs", "rank_mp", "rank_os", "rank_ov", "rank_ob"]

    for datasize in DataSize:
        for dataset in DataSet:
            if (dataset == "gist" and datasize != 1):
                continue
            for graph in Graph:
                program = "./main_" + graph
                command = program + " " + stage + " " + dataset + " " + str(datasize) + " " + str(threadsize)
                # os.system("cd build && make main && " + command)
                os.system("cd build && " + command)

# HNSW (prefetch) 多线程实验
def testHNSWMultiThread():
    DataSize = [10, 100]
    # DataSet = ["deep", "sift", "spacev"]
    DataSet = ["turing"]
    ThreadSize = []
    for i in range(0, 7):
        ThreadSize.append(2 ** i)

    stage = "search"
    graph = "hnsw"
    for datasize in DataSize:
        for dataset in DataSet:
            for ts in ThreadSize:
                if (dataset == "gist" and datasize != 1):
                    continue
                program = "./main_" + graph
                command = program + " " + stage + " " + dataset + " " + str(datasize) + " " + str(ts)
                os.system("cd build && " + command)

def testHNSW():
    stage = "search"

    # DataSet = ["deep", "sift", "spacev"]
    DataSet = ["turing"]
    DataSize = [1, 10, 100]
    Graph = ["hnsw"]
    threadsize = 1

    for datasize in DataSize:
        for dataset in DataSet:
            if (dataset == "gist" and datasize != 1):
                continue
            for graph in Graph:
                program = "./main_" + graph
                command = program + " " + stage + " " + dataset + " " + str(datasize) + " " + str(threadsize)
                os.system("cd build && " + command)

def main():
    testHNSWMultiThread()
    # testHNSW()

main()