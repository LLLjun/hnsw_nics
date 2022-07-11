import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


resultDir = 'output/result'

figDir = os.path.join(resultDir, 'figs')
if not os.path.exists(figDir):
    os.mkdir(figDir)



def loadData(filename):
    # efs Recall time_us
    data = pd.read_csv(filename, sep='\t')
    data.drop(['efs'], axis=1, inplace=True)
    return data

def setFig(fplt, savefile):
    fplt.grid(True)
    fplt.xlabel('R@10')
    fplt.ylabel('time (us)')
    fplt.savefig(savefile)


def plotFig(dataset, datasize, Rc = 10):
    Graph = ['hnsw', 'plat', 'hnsw_pf', 'plat_pf']
    unique = dataset + str(datasize) + 'm_rc' + str(Rc)
    logname = unique + '.log'

    # 显示创建figure对象
    fig = plt.figure(figsize=(5,5))
    for g in Graph:
        filepath = os.path.join(resultDir, g, logname)
        data = loadData(filepath)
        plt.plot(data['R@10'], data['time_us'], "-h", label=g)
        plt.legend()

    figname = unique + '.png'
    setFig(plt, os.path.join(figDir, figname))

def main():
    Datasize = [1, 10]
    Dataset = ['deep', 'sift', 'spacev', 'turing']

    for datasize in Datasize:
        for dataset in Dataset:
            plotFig(dataset, datasize)

main()