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
    plt.xlabel(columns[1])
    plt.ylabel(columns[2])

    nums = np_array.shape[0]
    for i in range(0, nums):
        label = legend_list[i]
        np_a = np_array[i]
        plt.semilogy(np_a[1], np_a[2], lw=3, label=label)
        
    plt.legend(loc='upper right', fontsize='large')

    fig_acc.savefig(path_fig, dpi=300)


def handle_data(dataname, efc_list, m_list, k, dms_list, ncf_list, reach_step_list):
    label = "expc2"
    datasize = 1

    columns = []
    # efc_list = [100, 300]
    # m_list = [30]
    # efc_list = range(100, 301, 100)
    # m_list = [20]

    data_list = []
    legend_list = []
    path_dataset = os.path.join(root_output, label, dataname)
    path_save = os.path.join(path_dataset, "fig")
    if os.path.exists(path_save) is False:
        os.mkdir(path_save)

    for efc in efc_list:
        for m in m_list:
            if efc > m:
                unique_name = dataname + str(datasize) + "m_ef" + str(efc) + "_M" + str(m) + "_k" + str(k) + "_search.csv"
                df_feature = pd.read_csv(os.path.join(path_dataset, unique_name))
                data_list.append(df_feature.values.transpose())
                legend_list.append(unique_name)

                for dms in dms_list:
                    for ncf in ncf_list:
                        for step in reach_step_list:
                            unique_name = dataname + str(datasize) + "m_ef" + str(efc) + "_M" + str(m) + "_dms" + str(dms) + "_ncf" + str(ncf) + "_reachstep" + str(step) + "_k" + str(k) + "_search_rldt_pro.csv"
                            df_feature = pd.read_csv(os.path.join(path_dataset, unique_name))
                            data_list.append(df_feature.values.transpose())
                            legend_list.append(unique_name)

                # unique_name = dataname + str(datasize) + "m_ef" + str(efc) + "_M" + str(m) + "_k" + str(k) + "_search_sxi.csv"
                # df_feature = pd.read_csv(os.path.join(path_dataset, unique_name))
                # data_list.append(df_feature.values.transpose())
                # legend_list.append(unique_name)

                # unique_name = dataname + str(datasize) + "m_ef" + str(efc) + "_M" + str(m) + "_k" + str(k) + "_search_sxi_rldt.csv"
                # df_feature = pd.read_csv(os.path.join(path_dataset, unique_name))
                # data_list.append(df_feature.values.transpose())
                # legend_list.append(unique_name)

                columns = df_feature.columns

    # efc_list = [200]
    # for efc in efc_list:
    #     for m in m_list:
    #         if efc > m:
    #             unique_name = dataname + str(datasize) + "m_ef" + str(efc) + "_M" + str(m) + "_k" + str(k) + "_search.csv"
    #             df_feature = pd.read_csv(os.path.join(path_dataset, unique_name))
    #             data_list.append(df_feature.values.transpose())
    #             legend_list.append(unique_name)

    np_feature = np.array(data_list).astype(np.float32)
    figname = "compare_rldt_pro_pp_" + dataname + str(datasize) + "m_k" + str(k)
    path_fig = os.path.join(path_save, figname)
    save_fig(path_fig, np_feature, legend_list, columns)

if __name__ == "__main__":
    handle_data()
