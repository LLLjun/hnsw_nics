import os
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


resultDir = 'output/result'


def loadData(filename, Rt = -1):
    # Rt Recall time_us
    data = pd.read_csv(filename, sep='\t')
    if (Rt != -1):
        data = data.loc[data['efs'] == Rt]
        # data.loc[0]
    data.drop(['efs'], axis=1, inplace=True)
    # print(data)
    # exit()
    return data

def setFig(fplt, savefile):
    fplt.grid(True)
    fplt.xlabel('R@10')
    fplt.ylabel('time (us)')
    fplt.savefig(savefile)

def setFigBar(fplt, savefile):
    fplt.grid(True)
    # fplt.xlabel('R@10')
    fplt.ylabel('time (us)')
    fplt.savefig(savefile)

def plotFig(figdir, dataset, datasize, ranksize, Rc = 10):
    hwconfig = 'rank_' + str(ranksize)
    unique = dataset + str(datasize) + 'm_rc' + str(Rc)

    # 显示创建figure对象
    fig = plt.figure(figsize=(5,5))

    Graph = ['plat']
    for g in Graph:
        logname = unique + '.log'
        filepath = os.path.join(resultDir, g, logname)
        data = loadData(filepath)
        plt.plot(data['R@10'], data['time_us'], "-h", label=g)
        plt.legend()

    Suffix = ['bs', 'mp', 'os', 'ov', 'ob']
    for sf in Suffix:
        logname = unique + '_' + sf + '.log'
        filepath = os.path.join(resultDir, hwconfig, logname)
        data = loadData(filepath)
        plt.plot(data['R@10'], data['time_us'], "-h", label=sf)
        plt.legend()

    figname = unique + '.png'
    setFig(plt, os.path.join(figdir, figname))

def plotBar(figdir, dataset, datasize, ranksize, Rt, Rc = 10):
    hwconfig = 'rank_' + str(ranksize)
    unique = dataset + str(datasize) + 'm_rc' + str(Rc)

    # 显示创建figure对象
    fig = plt.figure(figsize=(5,5))
    n_groups = 7
    index = 0
    width = 1 / n_groups;
    i_bar = 0

    plat_time = 0
    recall_rate = Rt
    Graph = ['plat']
    for g in Graph:
        logname = unique + '.log'
        filepath = os.path.join(resultDir, g, logname)
        data = loadData(filepath, Rt)
        plat_time = data['time_us']
        recall_rate = data['R@10']
        plt.bar(index + i_bar * width, 1, width, label=g)
        i_bar += 1
        plt.legend()

    Suffix = ['bs', 'mp', 'os', 'ov', 'ob']
    for sf in Suffix:
        logname = unique + '_' + sf + '.log'
        filepath = os.path.join(resultDir, hwconfig, logname)
        data = loadData(filepath, Rt)
        plt.bar(index + i_bar * width, (plat_time / data['time_us']), width, label=sf)
        i_bar += 1
        plt.legend()

    figname = unique + '.png'
    # plt.title('Recall@' + str(Rc) + ': ' + str(recall_rate))
    # plt.xticks(index, (Graph + Suffix))
    setFigBar(plt, os.path.join(figdir, figname))

def main():
    Datasize = [1, 10, 100]
    Dataset = ['deep', 'sift', 'spacev']
    Ranksize = 8

    # figDir = os.path.join(resultDir, ('figs_rank_' + str(Ranksize)))
    # if not os.path.exists(figDir):
    #     os.mkdir(figDir)

    # for datasize in Datasize:
    #     for dataset in Dataset:
    #         plotFig(figDir, dataset, datasize, Ranksize)

    figDir = os.path.join(resultDir, ('figs_speed_rank_' + str(Ranksize)))
    if not os.path.exists(figDir):
        os.mkdir(figDir)
    recall = 100

    for datasize in Datasize:
        for dataset in Dataset:
            plotBar(figDir, dataset, datasize, Ranksize, recall)

main()