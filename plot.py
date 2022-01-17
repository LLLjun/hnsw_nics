import pandas as pd
import numpy as np
import os
import shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
# plt.style.use('ggplot')

root_output = "/home/usr-xkIJigVq/vldb/hnsw_nics/output"


def save_fig(path_fig, np_array, legend_list, columns):
    fig_acc = plt.figure(figsize=(15, 15))

    # plt.ylim(0, 3)
    # plt.title(f'dist with {min_step} steps', size=20)
    plt.xlabel(columns[0])
    plt.ylabel(columns[1])

    nums = np_array.shape[0]
    for i in range(0, nums):
        label = legend_list[i]
        np_a = np_array[i]
        plt.semilogy(np_a[0], np_a[1], lw=3, label=label)
        
    plt.legend(loc='upper right', fontsize='large')

    fig_acc.savefig(path_fig, dpi=300)


def handle_data():
    label = "expc1"
    # datasets = ["turing"]
    datasets = ["deep", "sift", "turing"]
    datasize = 10
    # k = 1

    columns = []
    efc_list = []
    m_list = []
    k_list = [100]

    for dataname in datasets:
        path_dataset = os.path.join(root_output, label, dataname)
        path_save = os.path.join(path_dataset, "fig")
        if os.path.exists(path_save) is False:
            os.mkdir(path_save)

        if dataname == "gist":
            efc_list = range(80, 201, 40)
            m_list = [40]
        else:
            efc_list = range(60, 121, 20)
            m_list = [20]

        for k in k_list:
            data_list = []
            legend_list = []
            for efc in efc_list:
                for m in m_list:
                    if efc > m:
                        unique_name = dataname + str(datasize) + "m_ef" + str(efc) + "_M" + str(m) + "_k" + str(k) + "_search.csv"
                        df_feature = pd.read_csv(os.path.join(path_dataset, unique_name))
                        data_list.append(df_feature.values.transpose())
                        legend_list.append(unique_name)

                        unique_name = dataname + str(datasize) + "m_ef" + str(efc) + "_M" + str(m) + "_k" + str(k) + "_search_sxi.csv"
                        df_feature = pd.read_csv(os.path.join(path_dataset, unique_name))
                        data_list.append(df_feature.values.transpose())
                        legend_list.append(unique_name)

                        columns = df_feature.columns

            np_feature = np.array(data_list).astype(np.float32)
            figname = "compare_sxi_" + dataname + str(datasize) + "m_k" + str(k)
            path_fig = os.path.join(path_save, figname)
            save_fig(path_fig, np_feature, legend_list, columns)

if __name__ == "__main__":
    handle_data()
