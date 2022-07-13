import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


resultDir = 'output/result'


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


def plotFig(figdir, dataset, datasize, ranksize, Rc = 10):
    hwconfig = 'rank_' + str(ranksize)
    unique = dataset + str(datasize) + 'm_rc' + str(Rc)
    Suffix = ['bs', 'mp', 'os', 'ov', 'ob']

    # 显示创建figure对象
    fig = plt.figure(figsize=(5,5))
    for sf in Suffix:
        logname = unique + '_' + sf + '.log'
        filepath = os.path.join(resultDir, hwconfig, logname)
        data = loadData(filepath)
        plt.plot(data['R@10'], data['time_us'], "-h", label=sf)
        plt.legend()

    Graph = ['plat', 'plat_pf']
    for g in Graph:
        logname = unique + '.log'
        filepath = os.path.join(resultDir, g, logname)
        data = loadData(filepath)
        plt.plot(data['R@10'], data['time_us'], "-h", label=g)
        plt.legend()

    figname = unique + '.png'
    setFig(plt, os.path.join(figdir, figname))

def main():
    Datasize = [1, 10]
    Dataset = ['deep', 'sift', 'spacev']
    Ranksize = 8

    figDir = os.path.join(resultDir, ('figs_rank_' + str(Ranksize)))
    if not os.path.exists(figDir):
        os.mkdir(figDir)

    for datasize in Datasize:
        for dataset in Dataset:
            plotFig(figDir, dataset, datasize, Ranksize)

main()