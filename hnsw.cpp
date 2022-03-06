#include <iostream>
#include <fstream>
#include <queue>
#include <map>
#include <vector>
#include <chrono>
#include "hnswlib/hnswlib.h"
#include <unordered_set>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "omp.h"

using namespace std;
using namespace hnswlib;

template<typename DTres>
static void
get_gt(unsigned *massQA, size_t qsize, size_t &gt_maxnum, size_t vecdim,
        vector<std::priority_queue<std::pair<DTres, labeltype >>> &answers, size_t k) {
    (vector<std::priority_queue<std::pair<DTres, labeltype >>>(qsize)).swap(answers);
    cout << qsize << "\n";
    for (int i = 0; i < qsize; i++) {
        for (int j = 0; j < k; j++) {
            answers[i].emplace(0.0f, massQA[gt_maxnum * i + j]);
        }
    }
}

template<typename DTval, typename DTres>
static float
test_approx(DTval *massQ, size_t qsize, HierarchicalNSW<DTres> &appr_alg, size_t vecdim,
            vector<std::priority_queue<std::pair<DTres, labeltype >>> &answers, size_t k) {
    size_t correct = 0;
    size_t total = 0;
    //uncomment to test in parallel mode:
    //#pragma omp parallel for
    for (int i = 0; i < qsize; i++) {

        std::priority_queue<std::pair<DTres, labeltype >> result = appr_alg.searchKnn(massQ + vecdim * i, k);
        std::priority_queue<std::pair<DTres, labeltype >> gt(answers[i]);
        unordered_set<labeltype> g;
        total += gt.size();

        while (gt.size()) {
            g.insert(gt.top().second);
            gt.pop();
        }

        while (result.size()) {
            if (g.find(result.top().second) != g.end()) {
                correct++;
            } else {
            }
            result.pop();
        }
    }
    return 1.0f * correct / total;
}

template<typename DTval, typename DTres>
static void
test_vs_recall(DTval *massQ, size_t qsize, HierarchicalNSW<DTres> &appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<DTres, labeltype >>> &answers, size_t k) {
    vector<size_t> efs;// = { 10,10,10,10,10 };
    // for (int i = k; i < 30; i++) {
    //     efs.push_back(i);
    // }
    for (int i = 10; i < 100; i += 10) {
        efs.push_back(i);
    }
    // for (int i = 100; i < 500; i += 40) {
    //     efs.push_back(i);
    // }
    cout << "ef\t" << "R@" << k << "\t" << "qps\t" << "hop_0\t" << "hop_L\n";
    for (size_t ef : efs) {
        appr_alg.setEf(ef);
        appr_alg.metric_hops = 0;
        appr_alg.metric_hops_L = 0;
        clk_get stopw = clk_get();

        float recall = test_approx(massQ, qsize, appr_alg, vecdim, answers, k);
        float time_us_per_query = stopw.getElapsedTimeus() / qsize;
        float avg_hop_0 = 1.0f * appr_alg.metric_hops / qsize;
        float avg_hop_L = 1.0f * appr_alg.metric_hops_L / qsize;

        cout << ef << "\t" << recall << "\t" << 1e6 / time_us_per_query << "\t"
        << avg_hop_0 << "\t" << avg_hop_L << "\n";
        if (recall > 1.0) {
            cout << recall << "\t" << time_us_per_query << " us\n";
            break;
        }
    }
}

inline bool exists_test(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
}

/*
    将 base vectors 首先聚类为num_banks个类，然后根据 num_perspnode 决定是否需要二次聚类
    i.e.，num_perspnode = 1时仅采用DRAM
    input: base vectors
    output: mass_graph, (mass_globel)
*/
template<typename DTset, typename DTval, typename DTres>
void cluster_to_dram_ssd(const string &dataname, map<string, size_t> &index_parameter, map<string, string> &index_string){

    size_t vecsize = index_parameter["vecsize"];
    size_t vecdim = index_parameter["vecdim"];

    string path_data = index_string["path_data"];
    string dir_clu = index_string["dir_clu"];
    string format = index_string["format"];

    size_t num_banks = index_parameter["num_banks"];
    size_t num_perspnode = index_parameter["num_perspnode"];
    size_t vecsize_dram_total = vecsize / num_perspnode;
    size_t vecsize_dram_per_bank = vecsize_dram_total / num_banks;
    size_t vecsize_ssd_per_bank = vecsize / num_banks;

    string path_clu_graphId_to_externalId = index_string["path_clu_graphId_to_externalId"];
    string path_clu_mass_graph = index_string["path_clu_mass_graph"];
    string path_clu_mass_global = index_string["path_clu_mass_global"];
    string path_clu_globalId_to_externalId = index_string["path_clu_globalId_to_externalId"];


    DTset *massB = new DTset[vecsize * vecdim]();
    // ***
    // graphId_to_externalId.size = num_banks * vecsize_dram_per_bank
    unsigned *graphId_to_externalId = new unsigned[vecsize_dram_total]();
    // when storage, mass_bank for dram-only, mass_graph for dram-ssd
    DTset *mass_graph = new DTset[vecsize_dram_total * vecdim]();
    DTset **mass_bank = gene_array<DTset>(num_banks, vecsize_ssd_per_bank, vecdim);

    cout << "Loading base data:\n";
    if (format == "float"){
        LoadBinToArray<DTval>(path_data, massB, vecsize, vecdim);
    } else if (format == "uint8"){
        DTset *massB_int = new DTset[vecsize * vecdim]();
        LoadBinToArray<DTset>(path_data, massB_int, vecsize, vecdim);
        TransIntToFloat<DTset>(massB, massB_int, vecsize, vecdim);
        delete[] massB_int;
    } else {
        printf("Error, unsupport format \n");
        exit(1);
    }

    int num_iters = 20;
    L2Space l2_kmeans(vecdim);

    int *externalId_to_bankLabel = new int[vecsize]();
    int **bankLabel_to_externalId = gene_array<int>(num_banks, vecsize_ssd_per_bank);


#if NOCLUSTER
    if (num_perspnode != 1){
        printf("Error, unsuport.\n");
        exit(1);
    }

    std::vector<uint32_t> sample_list;
    for (uint32_t i = 0; i < vecsize; i++)
        sample_list.push_back(i);
    srand((unsigned int) time(NULL));
    random_shuffle(sample_list.begin(), sample_list.end());

    for (int i = 0; i < num_banks; i++){
        for (int j = 0; j < vecsize_dram_per_bank; j++){
            unsigned graph_id = i * vecsize_dram_per_bank + j;
            unsigned external_id = sample_list[graph_id];
            graphId_to_externalId[graph_id] = external_id;
            memcpy(mass_bank[i] + j * vecdim, massB + external_id * vecdim, vecdim * sizeof(DTset));
        }
    }
#else

    // vecsize -> num_banks * vecsize_ssd_per_bank
    K_means<DTset, DTres> *BankMeans = new K_means<DTset, DTres>(&l2_kmeans, vecdim);

    uint32_t num_sample = NUM_CLUSTER_TRAIN < vecsize ? NUM_CLUSTER_TRAIN : vecsize;
    DTset *mass_sample = new DTset[num_sample * vecdim]();
    std::vector<uint32_t> sample_list;
    for (uint32_t i = 0; i < vecsize; i++)
        sample_list.push_back(i);
    srand((unsigned int)time(NULL));
    random_shuffle(sample_list.begin(), sample_list.end());
    sample_list.resize(num_sample);
    std::vector<uint32_t>(sample_list).swap(sample_list);
    for (uint32_t i = 0; i < num_sample; i++){
        memcpy(mass_sample + i * vecdim, massB + sample_list[i] * vecdim, vecdim * sizeof(DTset));
    }
    printf("actual size: %u, sample size: %u \n", vecsize, num_sample);

    // 训练阶段为了精度，不设置上限
    BankMeans->train_cluster(num_banks, num_sample, num_sample, num_iters, sample_list, mass_sample);
    BankMeans->forward_cluster(vecsize, vecsize_ssd_per_bank, massB, externalId_to_bankLabel);
    BankMeans->~K_means();

    uint32_t *bank_pos = new uint32_t[num_banks]();
    for (size_t i = 0; i < vecsize; i++){
        uint32_t lb = externalId_to_bankLabel[i];
        uint32_t pos = bank_pos[lb];
        bankLabel_to_externalId[lb][pos] = i;
        bank_pos[lb]++;

        if (num_perspnode == 1)
            graphId_to_externalId[lb * vecsize_dram_per_bank + pos] = i;
        memcpy(mass_bank[lb] + pos * vecdim, massB + i * vecdim, vecdim * sizeof(DTset));
    }
    for (size_t i = 0; i < num_banks; i++){
        if (bank_pos[i] != vecsize_ssd_per_bank){
            printf("Error, cluster bank: %u, its num: %u, expect: %u \n", i, bank_pos[i], vecsize_ssd_per_bank);
            exit(1);
        }
    }

    delete[] mass_sample;
    delete[] massB;
    printf("Cluster base data: %u to bank level: %u * %u is done\n", vecsize, num_banks, vecsize_ssd_per_bank);
#endif

    if (mkdir(dir_clu.c_str(), S_IRWXU) != 0) {
        printf("Error, dir %s create failed \n", dir_clu.c_str());
        exit(1);
    }

    if (num_perspnode == 1){
        printf("Only-DRAM cluster...\n");

        WriteBinToArray<unsigned>(path_clu_graphId_to_externalId, graphId_to_externalId, num_banks, vecsize_dram_per_bank);
        for (int i = 0; i < num_banks; i++)
            memcpy(mass_graph + i * vecsize_dram_per_bank * vecdim,
                    mass_bank[i], vecsize_dram_per_bank * vecdim * sizeof(DTset));
        WriteBinToArray<DTset>(path_clu_mass_graph, mass_graph, vecsize_dram_total, vecdim);
        delete[] graphId_to_externalId;

    } else {
        printf("DRAM-SSD cluster...\n");

        // ***
        // globalId_to_externalId.size = vecsize_dram_total * num_perspnode
        unsigned *globalId_to_externalId = new unsigned[vecsize]();
        DTset *mass_global = new DTset[vecsize * vecdim]();

        // for (num_banks): vecsize_ssd_per_bank -> vecsize_dram_per_bank * num_perspnode
#pragma omp parallel for
        for (size_t bank_id = 0; bank_id < num_banks; bank_id++){
            // vecsize_dram_per_bank ~12.5M, double cluster
            vector<uint32_t> x;
            int **clu_to_inter = gene_array<int>(vecsize_dram_per_bank, num_perspnode);
            K_means<DTset, DTres> *GraphMeans = new K_means<DTset, DTres>(&l2_kmeans, vecdim);
            GraphMeans->train_cluster(vecsize_dram_per_bank, vecsize_ssd_per_bank, num_perspnode,
                                        num_iters, x, mass_bank[bank_id], false, (bank_id == 0));

            // get clu_to_inter
            uint32_t *clu_pos = new uint32_t[vecsize_dram_per_bank]();
            for (size_t j = 0; j < vecsize_ssd_per_bank; j++){
                uint32_t cl = GraphMeans->in_cluster[j];
                uint32_t pos = clu_pos[cl];
                clu_to_inter[cl][pos] = j;
                clu_pos[cl]++;
            }
            for (size_t i = 0; i < vecsize_dram_per_bank; i++){
                if (clu_pos[i] != num_perspnode){
                    printf("Error, cluster super node: %u, its num: %u, expect: %u \n",
                            (bank_id * vecsize_dram_per_bank + i), clu_pos[i], num_perspnode);
                    exit(1);
                }
            }

#pragma omp critical
{
            // write
            for (size_t bg_node_id = 0; bg_node_id < vecsize_dram_per_bank; bg_node_id++){
                size_t sp_node_id = bank_id * vecsize_dram_per_bank + bg_node_id;
                graphId_to_externalId[sp_node_id] = bankLabel_to_externalId[bank_id][clu_to_inter[bg_node_id][0]];
                memcpy(mass_graph + vecdim * sp_node_id,
                        mass_bank[bank_id] + vecdim * GraphMeans->cluster_center_id[bg_node_id], vecdim * sizeof(DTset));

                for (size_t i = 0; i < num_perspnode; i++){
                    globalId_to_externalId[sp_node_id * num_perspnode + i] =
                        bankLabel_to_externalId[bank_id][clu_to_inter[bg_node_id][i]];
                    memcpy(mass_global + vecdim * (sp_node_id * num_perspnode + i),
                            mass_bank[bank_id] + vecdim * clu_to_inter[bg_node_id][i], vecdim * sizeof(DTset));
                }
            }
            freearray<int>(clu_to_inter, vecsize_dram_per_bank);
            GraphMeans->~K_means();
}

        }

        printf("Cluster %u banks base data: %u to graph level: %u * %u is done\n",
                num_banks, vecsize_ssd_per_bank, vecsize_dram_per_bank, num_perspnode);

        WriteBinToArray<unsigned>(path_clu_graphId_to_externalId, graphId_to_externalId, num_banks, vecsize_dram_per_bank);
        WriteBinToArray<DTset>(path_clu_mass_graph, mass_graph, vecsize_dram_total, vecdim);
        WriteBinToArray<DTset>(path_clu_mass_global, mass_global, vecsize, vecdim);
        WriteBinToArray<unsigned>(path_clu_globalId_to_externalId, globalId_to_externalId, vecsize_dram_total, num_perspnode);
        delete[] graphId_to_externalId;
        delete[] mass_global;
        delete[] globalId_to_externalId;
    }

    delete[] mass_graph;
    freearray<DTset>(mass_bank, num_banks);
    freearray<int>(bankLabel_to_externalId, num_banks);
    printf("file in %s generate is done \n", dir_clu.c_str());
}

/*
    分别对 mass_graph vectors, mass_global vectors 和 query vectors 做定点量化
    input: mass_graph, mass_globel, query vectors
    output: mass_fix_graph, mass_fix_globel, mass_fix_Q_dram, mass_fix_Q_ssd
*/
template<typename DTset, typename DTval, typename DTres>
void fix_to_dram_ssd(const string &dataname, map<string, size_t> &index_parameter, map<string, string> &index_string){

    size_t vecsize = index_parameter["vecsize"];
    size_t vecdim = index_parameter["vecdim"];
    size_t qsize = index_parameter["qsize"];

    string path_q = index_string["path_q"];
    string dir_clu = index_string["dir_clu"];
    string dir_fix = index_string["dir_fix"];
    string format = index_string["format"];

    size_t num_banks = index_parameter["num_banks"];
    size_t num_perspnode = index_parameter["num_perspnode"];
    size_t vecsize_dram_total = vecsize / num_perspnode;
    size_t vecsize_dram_per_bank = vecsize_dram_total / num_banks;
    size_t vecsize_ssd_per_bank = vecsize / num_banks;

    string path_clu_graphId_to_externalId = index_string["path_clu_graphId_to_externalId"];
    string path_clu_mass_graph = index_string["path_clu_mass_graph"];
    string path_clu_mass_global = index_string["path_clu_mass_global"];
    string path_clu_globalId_to_externalId = index_string["path_clu_globalId_to_externalId"];

    string path_fix_mass_global = index_string["path_fix_mass_global"];
    string path_fix_mass_Q_ssd = index_string["path_fix_mass_Q_ssd"];
    string path_fix_mass_graph = index_string["path_fix_mass_graph"];
    string path_fix_mass_Q_dram = index_string["path_fix_mass_Q_dram"];
    string path_fix_flag_graph = index_string["path_fix_flag_graph"];
    string path_fix_flag_Q_dram = index_string["path_fix_flag_Q_dram"];


    DTset *massQ = new DTset[qsize * vecdim];
    cout << "Loading queries:\n";
    if (format == "float"){
        LoadBinToArray<DTval>(path_q, massQ, qsize, vecdim);
    } else if (format == "uint8"){
        DTset *massQ_int = new DTset[qsize * vecdim]();
        LoadBinToArray<DTset>(path_q, massQ_int, qsize, vecdim);
        TransIntToFloat<DTset>(massQ, massQ_int, qsize, vecdim);
        delete[] massQ_int;
    } else {
        printf("Error, unsupport format \n");
        exit(1);
    }

    printf("Quantization for DRAM, %d base data and %d queries.\n", vecsize_dram_total, qsize);
    DTFDRAM *mass_fix_graph = new DTFDRAM[vecsize_dram_total * vecdim]();
    DTFDRAM *mass_fix_Q_dram = new DTFDRAM[qsize * vecdim]();

    DTset *mass_graph = new DTset[vecsize_dram_total * vecdim]();
    LoadBinToArray<DTset>(path_clu_mass_graph, mass_graph, vecsize_dram_total, vecdim);

    DirectQuant<DTFDRAM> *QuantDRAM = new DirectQuant<DTFDRAM>(vecdim, true, true, true, vecsize_dram_total, qsize);
    QuantDRAM->FixDataPoint(mass_graph, vecsize_dram_total, vecsize_dram_total);
    QuantDRAM->AddFullDataToFix(mass_graph, mass_fix_graph, vecsize_dram_total, 0);
    QuantDRAM->AddFullDataToFix(massQ, mass_fix_Q_dram, qsize, 1);

    printf("Quantization error is %.3f, quantzation number is %d, overflow number is %d\n",
            QuantDRAM->_quant_err, QuantDRAM->_quant_nums, QuantDRAM->_overflow_nums);

    if (mkdir(dir_fix.c_str(), S_IRWXU) != 0) {
        printf("Error, dir %s create failed \n", dir_fix.c_str());
        exit(1);
    }
    WriteBinToArray<DTFDRAM>(path_fix_mass_graph, mass_fix_graph, vecsize_dram_total, vecdim);
    WriteBinToArray<DTFDRAM>(path_fix_mass_Q_dram, mass_fix_Q_dram, qsize, vecdim);
    WriteBinToArray<uint8_t>(path_fix_flag_graph, QuantDRAM->CoarseTableBase, vecsize_dram_total, QuantDRAM->_coarse_table_len);
    WriteBinToArray<uint8_t>(path_fix_flag_Q_dram, QuantDRAM->CoarseTableQuery, qsize, QuantDRAM->_coarse_table_len);

    delete[] mass_fix_graph;
    delete[] mass_fix_Q_dram;
    delete[] mass_graph;

    if (num_perspnode > 1){
        printf("Quantization for SSD, %d base data and %d queries.\n", vecsize, qsize);

        DTFSSD *mass_fix_global = new DTFSSD[vecsize * vecdim]();
        DTFSSD *mass_fix_Q_ssd = new DTFSSD[qsize * vecdim]();

        DTset *mass_global = new DTset[vecsize * vecdim]();
        LoadBinToArray<DTset>(path_clu_mass_global, mass_global, vecsize, vecdim);

        DirectQuant<DTFSSD> *QuantSSD = new DirectQuant<DTFSSD>(vecdim, false, true, true);
        QuantSSD->FixDataPoint(mass_global, vecsize, vecsize_dram_total);
        QuantSSD->AddFullDataToFix(mass_global, mass_fix_global, vecsize);
        QuantSSD->AddFullDataToFix(massQ, mass_fix_Q_ssd, qsize);

        printf("Quantization error is %.3f, quantzation number is %d, overflow number is %d\n",
                QuantSSD->_quant_err, QuantSSD->_quant_nums, QuantSSD->_overflow_nums);

        WriteBinToArray<DTFSSD>(path_fix_mass_global, mass_fix_global, vecsize, vecdim);
        WriteBinToArray<DTFSSD>(path_fix_mass_Q_ssd, mass_fix_Q_ssd, qsize, vecdim);

        delete[] mass_fix_global;
        delete[] mass_fix_Q_ssd;
        delete[] mass_global;
    }

    delete[] massQ;
    printf("file in %s generate is done \n\n", dir_fix.c_str());
}

/*
    针对 DRAM 层构建图索引
    input: mass_graph, mass_globel, query vectors
    output: mass_fix_graph, mass_fix_globel, mass_fix_Q_dram, mass_fix_Q_ssd
*/
template<typename DTset, typename DTval, typename DTres>
void construct_to_dram(const string &dataname, map<string, size_t> &index_parameter, map<string, string> &index_string, bool isSave){
    size_t efConstruction = index_parameter["efConstruction"];
    size_t M = index_parameter["M"];
    size_t vecsize = index_parameter["vecsize"];
    size_t vecdim = index_parameter["vecdim"];
    size_t qsize = index_parameter["qsize"];

    string dir_index = index_string["dir_index"];
    string format = index_string["format"];

    size_t num_banks = index_parameter["num_banks"];
    size_t num_perspnode = index_parameter["num_perspnode"];
    size_t vecsize_dram_total = vecsize / num_perspnode;
    size_t vecsize_dram_per_bank = vecsize_dram_total / num_banks;
    size_t vecsize_ssd_per_bank = vecsize / num_banks;

    string path_clu_graphId_to_externalId = index_string["path_clu_graphId_to_externalId"];
    string path_clu_mass_graph = index_string["path_clu_mass_graph"];
    string path_clu_mass_global = index_string["path_clu_mass_global"];
    string path_clu_globalId_to_externalId = index_string["path_clu_globalId_to_externalId"];

    string path_fix_mass_global = index_string["path_fix_mass_global"];
    string path_fix_mass_Q_ssd = index_string["path_fix_mass_Q_ssd"];
    string path_fix_mass_graph = index_string["path_fix_mass_graph"];
    string path_fix_mass_Q_dram = index_string["path_fix_mass_Q_dram"];
    string path_fix_flag_graph = index_string["path_fix_flag_graph"];
    string path_fix_flag_Q_dram = index_string["path_fix_flag_Q_dram"];

    // index set file path
    string path_index_prefix = dir_index + "/index_bank_";


    DTset *mass_graph = new DTset[vecsize_dram_total * vecdim]();
    LoadBinToArray<DTset>(path_clu_mass_graph, mass_graph, vecsize_dram_total, vecdim);
#if USEFIX
    DTFDRAM *mass_fix_graph = new DTFDRAM[vecsize_dram_total * vecdim]();
    uint8_t *flag_fix_graph = new uint8_t[vecsize_dram_total * (unsigned)ceil(vecdim / 8)]();
    LoadBinToArray<DTFDRAM>(path_fix_mass_graph, mass_fix_graph, vecsize_dram_total, vecdim);
    LoadBinToArray<uint8_t>(path_fix_flag_graph, flag_fix_graph, vecsize_dram_total, ceil(vecdim / 8));
#endif

    L2Space l2space(vecdim);
    std::vector<HierarchicalNSW<DTres> *> appr_alg(num_banks);
    for (size_t bank_i = 0; bank_i < num_banks; bank_i++){
        string path_index = path_index_prefix + to_string(bank_i) + ".bin";
        appr_alg[bank_i] = new HierarchicalNSW<DTres>(&l2space, vecsize_dram_per_bank, M, efConstruction);

        DTset *massbase = mass_graph + bank_i * vecsize_dram_per_bank * vecdim;
#if USEFIX
        DTFDRAM *massfix = mass_fix_graph + bank_i * vecsize_dram_per_bank * vecdim;
        uint8_t *flagfix = flag_fix_graph + bank_i * vecsize_dram_per_bank * (unsigned)ceil(vecdim / 8);
#endif

#if PLATG
        unsigned center_id = compArrayCenter<DTset>(massbase, vecsize_dram_per_bank, vecdim);
        appr_alg[bank_i]->addPoint((void *) (massbase + center_id * vecdim), (size_t) center_id);
        // todo: add flag
#else
        appr_alg[bank_i]->addPoint((void *) (massbase), (size_t) 0);
#endif
        cout << "Building index: " << bank_i << "\n";
        int j1 = 0;
        clk_get stopw = clk_get();
        clk_get stopw_full = clk_get();
        size_t report_every = vecsize_dram_per_bank / 10;
#pragma omp parallel for
        for (size_t i = 1; i < vecsize_dram_per_bank; i++) {
// #pragma omp critical
//                 {
//                     j1++;
//                     if (j1 % report_every == 0) {
//                         cout << j1 / (0.01 * vecsize_dram_per_bank) << " %, "
//                             << report_every / (1000.0 * stopw.getElapsedTimes()) << " kips " << " Mem: "
//                             << getCurrentRSS() / 1000000 << " Mb \n";
//                         stopw.reset();
//                     }
//                 }
#if PLATG
            size_t ic;
            if (i <= center_id)
                ic = i - 1;
            else
                ic = i;
            appr_alg[bank_i]->addPoint((void *) (massbase + ic * vecdim), ic);
#else
            appr_alg[bank_i]->addPoint((void *) (massbase + i * vecdim), i);
#endif
        }
        cout << "Build time:" << stopw_full.getElapsedTimes() << "  seconds\n";

#if USEFIX
        printf("Replace feature for index: %u \n", bank_i);
        appr_alg[bank_i]->ReplaceFeature(massfix, flagfix, vecsize_dram_per_bank);
#endif

        if (isSave)
            appr_alg[bank_i]->saveIndex(path_index);
    }
    delete[] mass_graph;
#if USEFIX
    delete[] mass_fix_graph;
    delete[] flag_fix_graph;
#endif

    printf("Build index in dir %s is succeed \n", dir_index.c_str());
}


// need to support:
// 分级量化，存储，均匀量化
// 建多个图，label不连续
// 对每个图进行量化，存储标志位
/*

*/


template<typename DTset, typename DTval, typename DTres>
void build_index(const string &dataname, map<string, size_t> &index_parameter, map<string, string> &index_string, bool isSave = true){

    size_t efConstruction = index_parameter["efConstruction"];
    size_t M = index_parameter["M"];
    size_t vecsize = index_parameter["vecsize"];
    size_t vecdim = index_parameter["vecdim"];
    size_t qsize = index_parameter["qsize"];

    string path_data = index_string["path_data"];
    string dir_clu = index_string["dir_clu"];
    string dir_fix = index_string["dir_fix"];
    string dir_index = index_string["dir_index"];

    size_t num_banks = index_parameter["num_banks"];
    size_t num_perspnode = index_parameter["num_perspnode"];
    size_t vecsize_dram_total = vecsize / num_perspnode;
    size_t vecsize_dram_per_bank = vecsize_dram_total / num_banks;
    size_t vecsize_ssd_per_bank = vecsize / num_banks;


    // Cluster Data
    if (!access(dir_clu.c_str(), R_OK|W_OK)){
        printf("dir %s is existed \n", dir_clu.c_str());
        // if (!(exists_test(path_clu_mass_graph) && exists_test(path_clu_mass_global) && exists_test(path_clu_globalId_to_externalId))){
        //     printf("Error, file no found \n");
        //     exit(1);
        // }
    } else {
        cluster_to_dram_ssd<DTset, DTval, DTres>(dataname, index_parameter, index_string);
    }

    // Fix Data
#if USEFIX
    if (!access(dir_fix.c_str(), R_OK|W_OK)){
        printf("dir %s is existed \n", dir_fix.c_str());
        // if (!(exists_test(path_fix_mass_global) && exists_test(path_fix_mass_Q_ssd) && exists_test(path_fix_mass_graph) &&
        //         exists_test(path_fix_mass_Q_dram) && exists_test(path_fix_flag_graph) && exists_test(path_fix_flag_Q_dram))){
        //     printf("Error, file no found \n");
        //     exit(1);
        // }
    } else {
        fix_to_dram_ssd<DTset, DTval, DTres>(dataname, index_parameter, index_string);
    }
#endif

    if (!access(dir_index.c_str(), R_OK|W_OK)){
        printf("Index set %s is existed \n", dir_index.c_str());
        return;
    } else {
        if (mkdir(dir_index.c_str(), S_IRWXU) != 0) {
            printf("Error, dir %s create failed \n", dir_index.c_str());
            exit(1);
        }
        construct_to_dram<DTset, DTval, DTres>(dataname, index_parameter, index_string, isSave);
    }
}

float evaluate_recall(unsigned *massQA, vector<vector<labeltype>> &result_final, size_t &qsize, size_t &k, size_t &gt_maxnum){
    size_t collect = 0;
    for (int qi = 0; qi < qsize; qi++){
        unordered_set<unsigned> gt;
        for (int i = 0; i < k; i++)
            gt.emplace(massQA[qi * gt_maxnum + i]);

        for (labeltype res: result_final[qi]){
            if (gt.find(res) != gt.end())
                collect++;
        }
    }

    return (float) collect / (qsize * k);
}

void evaluate_recall(unsigned *massQA, vector<vector<labeltype>> &result, size_t &k,
                    priority_queue<pair<size_t, int>> &ground_perbank, size_t &num_banks){

    unordered_set<unsigned> gt;
    for (int i = 0; i < k; i++)
        gt.emplace(massQA[i]);

    for (int i = 0; i < num_banks; i++){
        size_t collect = 0;
        for (labeltype res : result[i]){
            if (gt.find(res) != gt.end())
                collect++;
        }
        ground_perbank.emplace(collect, i);
    }
}

// need to support:
// 多个图搜索过程中的通信
// 多图结果排序
// SSD 暴力求解，排序

template<typename DTset, typename DTval, typename DTres>
void search_index(const string &dataname, map<string, size_t> &index_parameter, map<string, string> &index_string){
    //
    size_t k = index_parameter["k"];
    size_t vecsize = index_parameter["vecsize"];
    size_t qsize = index_parameter["qsize"];
    size_t vecdim = index_parameter["vecdim"];
    size_t gt_maxnum = index_parameter["gt_maxnum"];

    size_t num_banks = index_parameter["num_banks"];
    size_t num_perspnode = index_parameter["num_perspnode"];
    size_t vecsize_dram_total = vecsize / num_perspnode;
    size_t vecsize_dram_per_bank = vecsize_dram_total / num_banks;
    size_t vecsize_ssd_per_bank = vecsize / num_banks;

    string dir_clu = index_string["dir_clu"];
    string dir_fix = index_string["dir_fix"];
    string dir_index = index_string["dir_index"];
    string path_gt = index_string["path_gt"];

    string path_clu_graphId_to_externalId = index_string["path_clu_graphId_to_externalId"];
    string path_clu_globalId_to_externalId = index_string["path_clu_globalId_to_externalId"];
    string path_mass_global;

#if USEFIX
    path_mass_global = index_string["path_fix_mass_global"];
    string path_fix_mass_Q_ssd = index_string["path_fix_mass_Q_ssd"];
    string path_fix_mass_Q_dram = index_string["path_fix_mass_Q_dram"];
    string path_fix_flag_Q_dram = index_string["path_fix_flag_Q_dram"];
#else
    path_mass_global = index_string["path_clu_mass_global"];
    string path_q = index_string["path_q"];
    string format = index_string["format"];
#endif

    string path_index_prefix = index_string["path_index_prefix"];

    if (access(dir_index.c_str(), R_OK|W_OK)){
        printf("Error, index %s is unexisted \n", dir_index.c_str());
        exit(1);
    } else {

        unsigned *massQA = new unsigned[qsize * gt_maxnum]();
        cout << "Loading GT:\n";
        LoadBinToArray<unsigned>(path_gt, massQA, qsize, gt_maxnum);

#if USEFIX
        size_t flag_len = ceil(vecdim / 8);
        size_t len_per_query_comb = vecdim * sizeof(DTFDRAM) + flag_len;

        char *massQ = new char[qsize * len_per_query_comb]();
        DTFDRAM *mass_fix_Q_dram = new DTFDRAM[qsize * vecdim]();
        uint8_t *flag_fix_Q_dram = new uint8_t[qsize * flag_len]();

        cout << "Loading queries:\n";
        // concat vector and flag
        LoadBinToArray<DTFDRAM>(path_fix_mass_Q_dram, mass_fix_Q_dram, qsize, vecdim);
        LoadBinToArray<uint8_t>(path_fix_flag_Q_dram, flag_fix_Q_dram, qsize, flag_len);
        for (int i = 0; i < qsize; i++){
            memcpy(massQ + i * len_per_query_comb, mass_fix_Q_dram + i * vecdim, vecdim * sizeof(DTFDRAM));
            memcpy(massQ + i * len_per_query_comb + vecdim * sizeof(DTFDRAM),
                        flag_fix_Q_dram + i * flag_len, flag_len);
        }
        delete[] mass_fix_Q_dram;
        delete[] flag_fix_Q_dram;
        printf("Load fixed queries and dram flag done \n");

        // Load bank graph
        L2SpaceIntFlag l2if(vecdim);
        std::vector<HierarchicalNSW<FCP32> *> appr_alg(num_banks);
        for (size_t bank_i = 0; bank_i < num_banks; bank_i++){
            string path_index = path_index_prefix + to_string(bank_i) + ".bin";
            appr_alg[bank_i] = new HierarchicalNSW<FCP32>(&l2if, path_index, false);
        }

#else
        DTset *massQ = new DTset[qsize * vecdim];
        cout << "Loading queries:\n";
        if (format == "float"){
            LoadBinToArray<DTval>(path_q, massQ, qsize, vecdim);
        } else if (format == "uint8"){
            DTset *massQ_int = new DTset[qsize * vecdim]();
            LoadBinToArray<DTset>(path_q, massQ_int, qsize, vecdim);
            TransIntToFloat<DTset>(massQ, massQ_int, qsize, vecdim);
            delete[] massQ_int;
        } else {
            printf("Error, unsupport format \n");
            exit(1);
        }

        // Load bank graph
        L2Space l2float(vecdim);
        std::vector<HierarchicalNSW<DTres> *> appr_alg(num_banks);
        for (size_t bank_i = 0; bank_i < num_banks; bank_i++){
            string path_index = path_index_prefix + to_string(bank_i) + ".bin";
            appr_alg[bank_i] = new HierarchicalNSW<DTres>(&l2float, path_index, false);
        }
#endif

        unsigned *graphId_to_externalId = nullptr;
        unsigned *globalId_to_externalId = nullptr;
        DTFSSD *mass_fix_Q_ssd = nullptr;
        // process for dram-only or dram-ssd
        if (num_perspnode == 1){
            graphId_to_externalId = new unsigned[vecsize_dram_total]();
            LoadBinToArray<unsigned>(path_clu_graphId_to_externalId, graphId_to_externalId, num_banks, vecsize_dram_per_bank);
        } else {
#if USEFIX
            mass_fix_Q_ssd = new DTFSSD[qsize * vecdim]();
            LoadBinToArray<DTFSSD>(path_fix_mass_Q_ssd, mass_fix_Q_ssd, qsize, vecdim);
#endif
            globalId_to_externalId = new unsigned[vecsize]();
            LoadBinToArray<unsigned>(path_clu_globalId_to_externalId, globalId_to_externalId, vecsize_dram_total, num_perspnode);
        }

        printf("Search begin ... \n");
        cout << "efs_bank\t" << "R@" << to_string(k) << "\tNDC_avg" << "\tNDC_max" << "\ttime(us)" <<"\n";

        // size_t efs_bank = 150;
        vector<size_t> efs_bank_list;
        // efs_bank_list.push_back(90);
        for (size_t i = 10; i <= 90; i += 10)
            efs_bank_list.push_back(i);

        for (size_t efs_bank : efs_bank_list){
            vector<vector<labeltype>> result_final(qsize);
            vector<vector<vector<labeltype>>> result_bank(qsize);
            for (int i = 0; i < qsize; i++)
                result_bank[i].resize(num_banks);

            size_t k_dram;
            if (num_perspnode == 1)
                k_dram = k;
            else
                k_dram = num_banks * efs_bank * EFS_PROP;

            DTFSSD *mass_brute_ssd = nullptr;
            unsigned *id_brute_ssd = nullptr;
            if (num_perspnode > 1){
                mass_brute_ssd = new DTFSSD[k_dram * num_perspnode * vecdim]();
                id_brute_ssd = new unsigned[k_dram * num_perspnode]();
            }

            omp_set_num_threads(num_banks);
            for (int i = 0; i < num_banks; i++){
                appr_alg[i]->setEf(efs_bank);
                appr_alg[i]->metric_distance_computations = 0;
            }
            clk_get stop_full = clk_get();
            float time_search_total = 0;

            for (size_t q_i = 0; q_i < qsize; q_i++){
                // search stage 1: in DRAM
                // FCP32 thr_global = std::numeric_limits<FCP32>::max();
#if USEFIX
                std::vector<std::priority_queue<std::pair<FCP32, labeltype>>> return_bank(num_banks);
                std::priority_queue<std::pair<FCP32, labeltype>> result_dram;
                char *query_c = massQ + q_i * len_per_query_comb;
#else
                std::vector<std::priority_queue<std::pair<DTres, labeltype>>> return_bank(num_banks);
                std::priority_queue<std::pair<DTres, labeltype>> result_dram;
                DTset *query_c = massQ + q_i * vecdim;
#endif

                
// #pragma omp parallel for
                for (size_t bank_i = 0; bank_i < num_banks; bank_i++){
                    // appr_alg[bank_i]->setThr(&thr_global);

                    // todo
                    if (num_perspnode == 1)
                        return_bank[bank_i] = appr_alg[bank_i]->searchKnn(query_c, k);
                    else
                        return_bank[bank_i] = appr_alg[bank_i]->searchKnn(query_c, efs_bank);

                }

                // merge per bank's result
                for (size_t bank_i = 0; bank_i < num_banks; bank_i++){
                    while (!return_bank[bank_i].empty()){
                        unsigned graphId = bank_i * vecsize_dram_per_bank + return_bank[bank_i].top().second;
                        result_dram.emplace(std::make_pair(return_bank[bank_i].top().first, graphId));
                        while(result_dram.size() > k_dram)
                            result_dram.pop();

                        // if (num_perspnode == 1 && return_bank[bank_i].size() <= k)
                        //     result_bank[q_i][bank_i].push_back(graphId_to_externalId[graphId]);

                        return_bank[bank_i].pop();
                    }
                }

                if (num_perspnode == 1){
                    while (!result_dram.empty()){
                        if (result_dram.size() <= k)
                            result_final[q_i].push_back(graphId_to_externalId[result_dram.top().second]);
                        result_dram.pop();
                    }

                } else {
                    // search in SSD
                    size_t efs_real = std::min(k_dram, result_dram.size());
                    memset(mass_brute_ssd, 0, k_dram * num_perspnode * vecdim * sizeof(DTFSSD));
                    memset(id_brute_ssd, 0, k_dram * num_perspnode * sizeof(unsigned));

                    ifstream inputBS(path_mass_global.c_str(), ios::binary);
                    uint32_t nums_r, dims_r;
                    inputBS.read((char *) &nums_r, sizeof(uint32_t));
                    inputBS.read((char *) &dims_r, sizeof(uint32_t));
                    if ((vecsize != nums_r) || (vecdim != dims_r)){
                        printf("Error, file size is error, nums_r: %u, dims_r: %u\n", nums_r, dims_r);
                        exit(1);
                    }
                    for (size_t i = 0; i < efs_real; i++){
                        size_t pos_r = result_dram.top().second;
                        inputBS.seekg(pos_r * num_perspnode * vecdim * sizeof(DTFSSD) + 2 * sizeof(unsigned), ios::beg);
                        inputBS.read((char *)(mass_brute_ssd + i * num_perspnode * vecdim), num_perspnode * vecdim * sizeof(DTFSSD));
                        result_dram.pop();
                        memcpy(id_brute_ssd + i * num_perspnode, globalId_to_externalId + pos_r * num_perspnode, num_perspnode * sizeof(unsigned));
                    }
                    inputBS.close();

#if USEFIX
                    L2SpaceSSD l2ssd(vecdim);
                    BruteforceSearch<FCP64>* brute_alg = new BruteforceSearch<FCP64>(&l2ssd, (size_t)(efs_real * num_perspnode));
#else
                    L2Space l2ssd(vecdim);
                    BruteforceSearch<DTres>* brute_alg = new BruteforceSearch<DTres>(&l2ssd, (size_t)(efs_real * num_perspnode));
#endif

                    omp_set_num_threads(omp_get_num_procs());
        // #pragma omp parallel for
                    for (size_t i = 0; i < (efs_real * num_perspnode); i++){
                        brute_alg->addPoint((void *) (mass_brute_ssd + i * vecdim), (size_t) id_brute_ssd[i]);
                        // for (size_t j = 0; j < gt_maxnum; j++){
                        //     if (massQA[q_i * gt_maxnum + j] == id_brute_ssd[i]){
                        //         printf("%u: %u\n", massQA[q_i * gt_maxnum + j], i);
                        //         break;
                        //     }
                        // }
                    }

#if USEFIX
                    std::priority_queue<std::pair<FCP64, labeltype >> rs = brute_alg->searchKnn(mass_fix_Q_ssd + q_i * vecdim, k);
#else
                    std::priority_queue<std::pair<DTres, labeltype >> rs = brute_alg->searchKnn(mass_fix_Q_ssd + q_i * vecdim, k);
#endif
                    while (!rs.empty()){
                        result_final[q_i].push_back(rs.top().second);
                        rs.pop();
                    }
                }
            }
            float time_per_query = stop_full.getElapsedTimeus() / qsize;
            // time_per_query = time_search_total / qsize;

            size_t NDC = 0;
            size_t NDC_maxbank = 0;
            for (int i = 0; i < num_banks; i++){
                NDC += appr_alg[i]->metric_distance_computations;
                NDC_maxbank = max<size_t>(NDC_maxbank, appr_alg[i]->metric_distance_computations);
            }
            float recall = evaluate_recall(massQA, result_final, qsize, k, gt_maxnum);
            cout << efs_bank << "\t" << recall << "\t" << (float) NDC / qsize << "\t"
                 << (float) NDC_maxbank / qsize << "\t" << time_per_query << "\n";

            // 测试不同bank对召回率的贡献度
            // vector<priority_queue<pair<size_t, int>>> recall_bank(qsize);
            // for (size_t q_i = 0; q_i < qsize; q_i++)
            //     evaluate_recall(massQA + q_i * gt_maxnum, result_bank[q_i], k, recall_bank[q_i], num_banks);

            // for (int i = 0; i < num_banks; i++){
            //     size_t collect = 0;
            //     vector<int> prop_bank(num_banks, 0);

            //     for (size_t q_i = 0; q_i < qsize; q_i++){
            //         size_t num_ground = recall_bank[q_i].top().first;
            //         collect += num_ground;
            //         if (num_ground)
            //             prop_bank[recall_bank[q_i].top().second]++;
            //         recall_bank[q_i].pop();
            //     }
            //     printf("top-%d, r = %.4f \t", (i + 1), (float) collect / (qsize * k));
            //     for (int i = 0; i < num_banks; i++)
            //         printf("%.3f\t", (float) prop_bank[i] / qsize);
            //     printf("\n");
            // }


        }

        printf("Search index %s is succeed \n", dir_index.c_str());
    }
}

// need to support:
//

void hnsw_impl(bool is_build, const string &using_dataset){
    string path_project = "/home/usr-xkIJigVq/nmp/hnsw_nics";

    string label = "dram-only";
    string path_graphindex = path_project + "/graphindex/" + label;

	size_t subset_size_milllions = 1;
	size_t efConstruction = 200;
	size_t M = 20;
    size_t k = 10;

    size_t vecsize = subset_size_milllions * 1000000;
    size_t qsize, vecdim, gt_maxnum;
    string path_index, path_gt, path_q, path_data;


    size_t num_banks = NUM_BANKS;
    size_t num_perspnode = NUM_PERSPNODE;

    std::map<string, size_t> index_parameter;
    index_parameter["subset_size_milllions"] = subset_size_milllions;
    index_parameter["efConstruction"] = efConstruction;
    index_parameter["M"] = M;
    index_parameter["k"] = k;
    index_parameter["vecsize"] = vecsize;
    index_parameter["num_banks"] = num_banks;
    index_parameter["num_perspnode"] = num_perspnode;

    std::map<string, string> index_string;
    index_string["format"] = "float";

    /*
    file structure
    --nmp
       |--deep10m_bk8pspn10
            |--clu_data
            |--fix_data
            |--index
    */
    string unique_name = using_dataset + to_string(subset_size_milllions) +
                        "m_bk" + to_string(num_banks) + "spn" + to_string(num_perspnode);
    string dir_this = path_graphindex + "/" + unique_name;
    if (access(dir_this.c_str(), R_OK|W_OK)){
        if (mkdir(dir_this.c_str(), S_IRWXU) != 0) {
            printf("Error, dir %s create failed \n", dir_this.c_str());
            exit(1);
        }
    }

    index_string["dir_clu"] = dir_this + "/clu_data";
    index_string["dir_fix"] = dir_this + "/fix_data";
#if USEFIX
    index_string["dir_index"] = dir_this + "/fix_ef" + to_string(efConstruction) + "m" + to_string(M);
#else
    index_string["dir_index"] = dir_this + "/float_ef" + to_string(efConstruction) + "m" + to_string(M);
#endif

    CheckDataset(using_dataset, index_parameter, index_string);
    SetPathStr(index_string);

    // index_string["gt_id"] = "/home/ljun/anns/hnsw_nics/graphindex/brute/deep/deep100k_gt_id.bin";
    // index_string["gt_dist"] = "/home/ljun/anns/hnsw_nics/graphindex/brute/deep/deep100k_gt_dist.bin";
    // index_parameter["gt_maxnum"] = 10;

    if (is_build){
        build_index<DTSET, DTVAL, DTRES>(using_dataset, index_parameter, index_string);
    } else{
        search_index<DTSET, DTVAL, DTRES>(using_dataset, index_parameter, index_string);
    }
    return;
}
