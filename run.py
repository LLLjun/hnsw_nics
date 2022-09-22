import os
import sys

class RunGraph:
    def __init__(self, stage, datasize, dataset):
        self.stage = stage
        self.DataSize = datasize
        self.DataSet = dataset
        self.Graph = []
        self.ThreadSize = []

    def searchGraphIndex(self):
        if self.stage != "search":
            sys.exit()

        for datasize in self.DataSize:
            for dataset in self.DataSet:
                if (dataset == "gist" and datasize != 1):
                    continue
                for graph in self.Graph:
                    for ts in self.ThreadSize:
                        program = "./main_" + graph
                        command = program + " " + self.stage + " " + dataset + " " + str(datasize) + " " + str(ts)
                        os.system("cd build && " + command)

    def testRank(self):
        self.Graph = ["rank_bs", "rank_mp", "rank_os", "rank_ov", "rank_ob"]
        self.ThreadSize = [1]
        self.searchGraphIndex()

    def testMultiGraph(self):
        self.Graph = ["hnsw", "plat"]
        self.ThreadSize = [1]
        self.searchGraphIndex()

    # HNSW (prefetch) 多线程实验
    def testHNSWMultiThread(self):
        self.Graph = ["hnsw"]
        self.ThreadSize = []
        for i in range(0, 7):
            self.ThreadSize.append(2 ** i)
        self.searchGraphIndex()

    def testHNSW(self):
        self.Graph = ["hnsw"]
        self.ThreadSize = [1]
        self.searchGraphIndex()


def main():
    stage = "search"
    Datasets = ["sift", "spacev"]
    Datasize = [1000]
    Threads = [1]
    sub_graph = 8

    os.system("cd build && make main")
    for datasize in Datasize:
        for dataset in Datasets:
            for thread in Threads:
                command = "./main " + stage + " " + dataset + " " + str(datasize) + " " + str(thread) + " " + str(sub_graph)
                os.system("cd build && " + command)

    # run = RunGraph(stage, Datasize, Datasets)
    # run.testMultiGraph()

main()