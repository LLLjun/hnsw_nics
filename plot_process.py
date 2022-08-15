import os
import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DEBUG = False

class Process:
    debug = DEBUG
    resultDir = 'output/result'
    figureDir = 'output/figure'

    dictQuerySize = {
        "deep": 10000,
        "sift": 10000,
        "spacev": 23000
    }

    # figure
    label_size = 15
    title_size = 15

    def __init__(self, topk):
        self.topk = topk
        self.str_recall = 'R@' + str(topk)
        self.str_time = 'time_us'
        self.str_QPS = 'QPS'

    def loadData(self, datafile_name):
        data = pd.read_csv(datafile_name, sep='\t')
        return data

    def addQPS(self, ds_name, data):
        data[self.str_QPS] = 1e6 / data[self.str_time]
        return data

    def plotInit(self, name):
        figdir_name = os.path.join(self.figureDir, name)
        if not os.path.exists(figdir_name):
            os.mkdir(figdir_name)
        return figdir_name

    # 根据指定的recall要求，返回最接近的idx
    def getIdxByRecall(self, datafile_name, recall_rate):
        data = self.loadData(datafile_name)
        data[self.str_recall] = (data[self.str_recall] - recall_rate).abs()
        idxmin = data[self.str_recall].idxmin()
        rate_error = data[self.str_recall].min()
        if rate_error > 0.005:
            print("Error, target rate: %.3f, rate error: %.3f, file name: %s \n"%(recall_rate, rate_error, datafile_name))
            sys.exit()
        return idxmin, data.loc[idxmin, self.str_time]

    # 读取多个文件的 time 值
    def getListByIdx(self, idx, file_prefix, file_range):
        values = []
        for v in file_range:
            file_name = file_prefix + str(v) + '.log'
            data = self.loadData(file_name)
            values.append(data.loc[idx, self.str_time])
        return values

    def plotBar(self, fig_name, labels, values):
        # 显示创建figure对象
        fig = plt.figure(figsize=(5,5))
        n_groups = values.shape[0]
        index = 0
        width = 1 / n_groups
        i_bar = 0

        for i in range(n_groups):
            plt.bar(index + i_bar * width, values[i], width, label=labels[i])
            i_bar += 1
            plt.legend()

        fig.grid(True)
        fig.ylabel('Normalized throughput')
        fig.savefig(fig_name)

    # 在给定的recall下，不同配置（这里是线程）之间的相对QPS比值
    def plotMultiThreadQPS(self, recall_rate):
        Datasize = [10, 100]
        Dataset = ['sift', 'deep', 'spacev']

        figdir_name = self.plotInit('HNSW_MultiThread')

        for datasize in Datasize:
            # 把相同大小的不同数据集画在一起
            num_dataset = len(Dataset)
            fig, ax = plt.subplots(1, num_dataset, figsize=(15, 5), sharex=True, sharey=True)

            for ds_i, dataset in enumerate(Dataset):
                unique_name = dataset + str(datasize) + 'm_rc' + str(self.topk)
                # basefile_name = os.path.join(self.resultDir, 'hnsw_pf', unique_name + '.log')
                basefile_name = os.path.join(self.resultDir, 'hnsw_pf_multithread', unique_name + '_t1.log')
                otherfile_prefix = os.path.join(self.resultDir, 'hnsw_pf_multithread', unique_name + '_t')
                range_list = []
                for i in range(1, 7):
                    range_list.append(2 ** i)

                idx, time_base = self.getIdxByRecall(basefile_name, recall_rate)
                values = self.getListByIdx(idx, otherfile_prefix, range_list)
                values.insert(0, time_base)
                labels = range_list
                labels.insert(0, 1)

                values = np.array(values, dtype=np.float32).reshape(-1, 1)

                if self.debug:
                    print(values)
                    print(labels)
                # values 归一化成吞吐量
                values = values[0] / values

                ax[ds_i].plot(labels, values, '-h', linestyle='dashed')
                ax[ds_i].plot(labels, labels, linestyle='dashed')
                ax[ds_i].grid(True)
                subtitle = dataset + str(datasize) + 'm'
                ax[ds_i].set_title(subtitle, fontsize=self.title_size)
                ax[ds_i].tick_params(labelsize=self.label_size-2)
                ax[ds_i].set_xlabel('Threads', fontsize=self.label_size)
                if ds_i == 0:
                    ax[ds_i].set_ylabel('Normalized throughput', fontsize=self.label_size)

            suptitle = 'Throughput of HNSW on multithreading (Recall@' + str(self.topk) + '=' + str(recall_rate) + ')'
            plt.suptitle(suptitle, fontsize=self.title_size)
            # tight_layout调整大小的时候没有考虑suptitle，因此需要手动设置范围避免重叠
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            figfile_name = os.path.join(figdir_name, 'multithread_' + str(datasize) + 'm_' + str(recall_rate) + '.png')
            plt.savefig(figfile_name)

    # QPS(Latency) - Recall 曲线
    def plotQPSToRecall(self):
        Datasize = [1, 10, 100]
        Dataset = ['sift', 'deep', 'spacev']
        Graph = ['hnsw_pf', 'plat_pf']

        figdir_name = self.plotInit('HNSW_Baseline')
        Performance = ["QPS", "Latency"]

        for datasize in Datasize:
            for perf in Performance:
                # 把相同大小的不同数据集画在一起
                num_dataset = len(Dataset)
                fig, ax = plt.subplots(1, num_dataset, figsize=(15, 5), sharex=True, sharey=True)
                if perf == "QPS":
                    for axi in ax:
                        axi.set_yscale('log')
                    ax[0].set_ylabel(perf, fontsize=self.label_size)
                else:
                    ax[0].set_ylabel(perf + ' (us)', fontsize=self.label_size)

                for ds_i, dataset in enumerate(Dataset):
                    unique_name = dataset + str(datasize) + 'm_rc' + str(self.topk)
                    for g in Graph:
                        datafile_name = os.path.join(self.resultDir, g, unique_name + '.log')
                        data = self.loadData(datafile_name)
                        if perf == "QPS":
                            data = self.addQPS(dataset, data)
                            ax[ds_i].plot(data[self.str_recall], data[self.str_QPS], '-h', label=g)
                        else:
                            ax[ds_i].plot(data[self.str_recall], data[self.str_time], '-h', label=g)
                    ax[ds_i].legend(fontsize=self.label_size)
                    ax[ds_i].grid(True)
                    subtitle = dataset + str(datasize) + 'm'
                    ax[ds_i].set_title(subtitle, fontsize=self.title_size)
                    ax[ds_i].tick_params(labelsize=self.label_size-2)
                    ax[ds_i].set_xlabel(self.str_recall, fontsize=self.label_size)

                suptitle = 'Performance'
                plt.suptitle(suptitle, fontsize=self.title_size)
                # tight_layout调整大小的时候没有考虑suptitle，因此需要手动设置范围避免重叠
                plt.tight_layout(rect=[0, 0, 1, 0.95])
                figfile_name = os.path.join(figdir_name, perf + '_' + str(datasize) + 'm.png')
                plt.savefig(figfile_name)

def main():
    topk = 10
    process = Process(topk)
    # process.plotMultiThreadQPS(0.95)
    process.plotQPSToRecall()

main()