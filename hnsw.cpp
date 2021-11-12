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

// need to support:
// 分级量化，存储，均匀量化
// 建多个图，label不连续
// 对每个图进行量化，存储标志位
/*

*/


template<typename DTset, typename DTval, typename DTres>
void build_index(const string &dataname, map<string, size_t> &index_parameter, map<string, string> &index_string, bool isSave = true){

    size_t efConstruction = index_parameter["efConstruction"];
    // size_t efConstruction = (index_parameter.find("efConstruction"))->second;
    size_t M = index_parameter["M"];
    size_t vecsize = index_parameter["vecsize"];
    size_t vecdim = index_parameter["vecdim"];
    size_t qsize = index_parameter["qsize"];
    
    string path_data = index_string["path_data"];
    // string path_data = (index_string.find("path_data"))->second;
    string path_q = index_string["path_q"];
    string dir_clu = index_string["dir_clu"];
    string dir_fix = index_string["dir_fix"];
    string dir_index = index_string["dir_index"];

    size_t num_banks = index_parameter["num_banks"];
    size_t num_perspnode = index_parameter["num_perspnode"];
    size_t vecsize_dram_total = vecsize / num_perspnode;
    size_t vecsize_dram_per_bank = vecsize_dram_total / num_banks;
    size_t vecsize_ssd_per_bank = vecsize / num_banks;

    // cluster data file path
    string path_clu_mass_graph = dir_clu + "/mass_graph.bin";
    string path_clu_mass_global = dir_clu + "/mass_global.bin";
    string path_clu_global_label_to_id = dir_clu + "/global_label_to_id.bin";
    // quantization data file path
    string path_fix_mass_global = dir_fix + "/mass_global.bin";
    string path_fix_mass_Q_ssd = dir_fix + "/mass_Q_ssd.bin";
    string path_fix_mass_graph = dir_fix + "/mass_graph.bin";
    string path_fix_mass_Q_dram = dir_fix + "/mass_Q_dram.bin";
    string path_fix_flag_graph = dir_fix + "/flag_graph.bin";
    string path_fix_flag_Q_dram = dir_fix + "/flag_Q_dram.bin";
    // index set file path
    string path_index_prefix = dir_index + "/index_bank_";

    // Cluster Data
    if (!access(dir_clu.c_str(), R_OK|W_OK)){
        printf("dir %s is existed \n", dir_clu.c_str());
        if (!(exists_test(path_clu_mass_graph) && exists_test(path_clu_mass_global) && exists_test(path_clu_global_label_to_id))){
            printf("Error, file no found \n");
            exit(1);
        }
    } else {


        DTset *massB = new DTset[vecsize * vecdim]();

        cout << "Loading base data:\n";
        ifstream inputB(path_data.c_str(), ios::binary);
        for (size_t i = 0; i < vecsize; i++){
            int expect_in;
            if (dataname == "sift" || dataname == "gist" || dataname == "deep"){
                inputB.read((char *) &expect_in, 4);
                if (expect_in != vecdim) {
                    cout << "file error \n";
                    exit(1);
                }
            }
            inputB.read((char *) (massB + i * vecdim), vecdim * sizeof(DTset));
        }
        inputB.close();

        int num_iters = 20;
        L2Space l2_kmeans(vecdim);

        int *id_to_bank_label = new int[vecsize]();
        int **bank_label_to_id = gene_array<int>(num_banks, vecsize_ssd_per_bank);
        DTset **mass_bank = gene_array<DTset>(num_banks, vecsize_ssd_per_bank, vecdim);
        
        // vecsize -> num_banks * vecsize_ssd_per_bank
        K_means<DTset, DTres> *BankMeans = new K_means<DTset, DTres>(&l2_kmeans, vecdim);
        uint32_t *bank_pos = new uint32_t[num_banks]();
        uint32_t num_sample = NUM_CLUSTER_TRAIN;
        DTset *mass_sample = new DTset[num_sample * vecdim]();

        std::vector<uint32_t> sample_list;
        for (uint32_t i = 0; i < vecsize; i++)
            sample_list.push_back(i);
        random_shuffle(sample_list.begin(), sample_list.end());
        sample_list.resize(num_sample);
        std::vector<uint32_t>(sample_list).swap(sample_list);
        for (uint32_t i = 0; i < num_sample; i++){
            memcpy(mass_sample + i * vecdim, massB + sample_list[i] * vecdim, vecdim * sizeof(DTset));
        }

        BankMeans->train_cluster(num_banks, num_sample, num_sample, num_iters, sample_list, mass_sample);
        BankMeans->forward_cluster(vecsize, vecsize_ssd_per_bank, massB, id_to_bank_label);
        BankMeans->~K_means();
        for (size_t i = 0; i < vecsize; i++){
            uint32_t lb = id_to_bank_label[i];
            uint32_t pos = bank_pos[lb];
            bank_label_to_id[lb][pos] = i;
            memcpy(mass_bank[lb] + pos * vecdim, massB + i * vecdim, vecdim * sizeof(DTset));
            bank_pos[lb]++;
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

        // unsigned *graph_id_to_label = new unsigned[vecsize_dram_total]();
        // unsigned *graph_label_to_id = new unsigned[vecsize_dram_total]();
        DTset *mass_graph = new DTset[vecsize_dram_total * vecdim]();

        // unsigned *global_id_to_label = new unsigned[vecsize]();
        // [vecsize_dram_total, num_perspnode]
        unsigned *global_label_to_id = new unsigned[vecsize]();
        DTset *mass_global = new DTset[vecsize * vecdim]();

        // for (num_banks): vecsize_ssd_per_bank -> vecsize_dram_per_bank * num_perspnode
// #pragma omp parallel for
        vector<uint32_t> x;
        for (size_t bank_id = 0; bank_id < num_banks; bank_id++){
            // vecsize_dram_per_bank ~12.5M, double cluster
            int **clu_to_inter = gene_array<int>(vecsize_dram_per_bank, num_perspnode);
            K_means<DTset, DTres> *GraphMeans = new K_means<DTset, DTres>(&l2_kmeans, vecdim);
            GraphMeans->train_cluster(vecsize_dram_per_bank, vecsize_ssd_per_bank, num_perspnode,
                                        num_iters, x, mass_bank[bank_id], false);
            
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

            // write
            for (size_t bg_node_id = 0; bg_node_id < vecsize_dram_per_bank; bg_node_id++){
                size_t sp_node_id = bank_id * vecsize_dram_per_bank + bg_node_id;
                memcpy(mass_graph + vecdim * sp_node_id, 
                        mass_bank[bank_id] + vecdim * GraphMeans->cluster_center_id[bg_node_id], vecdim * sizeof(DTset));

                for (size_t i = 0; i < num_perspnode; i++){
                    global_label_to_id[sp_node_id * num_perspnode + i] = 
                        bank_label_to_id[bank_id][clu_to_inter[bg_node_id][i]];
                    memcpy(mass_global + vecdim * (sp_node_id * num_perspnode + i), 
                            mass_bank[bank_id] + vecdim * clu_to_inter[bg_node_id][i], vecdim * sizeof(DTset));
                }
            }
            freearray<int>(clu_to_inter, vecsize_dram_per_bank);
            GraphMeans->~K_means();

        }
        delete[] mass_bank;
        freearray<int>(bank_label_to_id, num_banks);
        printf("Cluster %u banks base data: %u to graph level: %u * %u is done\n", 
                num_banks, vecsize_ssd_per_bank, vecsize_dram_per_bank, num_perspnode);

        if (mkdir(dir_clu.c_str(), S_IRWXU) != 0) {
            printf("Error, dir %s create failed \n", dir_clu.c_str());
            exit(1);
        }
        WriteBinToArray<DTset>(path_clu_mass_graph, mass_graph, vecsize_dram_total, vecdim);
        WriteBinToArray<DTset>(path_clu_mass_global, mass_global, vecsize, vecdim);
        WriteBinToArray<unsigned>(path_clu_global_label_to_id, global_label_to_id, vecsize_dram_total, num_perspnode);
        delete[] mass_graph;
        delete[] mass_global;
        delete[] global_label_to_id;

        printf("file in %s generate is done \n", dir_clu.c_str());
    }

    // Fix Data
    if (!access(dir_fix.c_str(), R_OK|W_OK)){
        printf("dir %s is existed \n", dir_fix.c_str());
        if (!(exists_test(path_fix_mass_global) && exists_test(path_fix_mass_Q_ssd) && exists_test(path_fix_mass_graph) &&
                exists_test(path_fix_mass_Q_dram) && exists_test(path_fix_flag_graph) && exists_test(path_fix_flag_Q_dram))){
            printf("Error, file no found \n");
            exit(1);
        }
    } else {


        DTset *massQ = new DTset[qsize * vecdim];
        DTset *mass_graph = new DTset[vecsize_dram_total * vecdim]();
        DTset *mass_global = new DTset[vecsize * vecdim]();
        LoadBinToArray<DTset>(path_clu_mass_graph, mass_graph, vecsize_dram_total, vecdim);
        LoadBinToArray<DTset>(path_clu_mass_global, mass_global, vecsize, vecdim);

        cout << "Loading queries:\n";
        ifstream inputQ(path_q.c_str(), ios::binary);
        for (int i = 0; i < qsize; i++) {
            if (dataname == "sift" || dataname == "gist" || dataname == "deep"){
                int in = 0;
                inputQ.read((char *) &in, 4);
                if (in != vecdim) {
                    cout << "file error" << vecdim << endl;
                    exit(1);
                }
            }
            // glove的queries没有维度信息
            inputQ.read((char *) (massQ + i * vecdim), vecdim * sizeof(DTset));
        }
        inputQ.close();
        printf("Load queries from %s done \n", path_q.c_str());

        DTFSSD *mass_fix_global = new DTFSSD[vecsize * vecdim]();
        DTFDRAM *mass_fix_graph = new DTFDRAM[vecsize_dram_total * vecdim]();
        DTFSSD *mass_fix_Q_ssd = new DTFSSD[qsize * vecdim]();
        DTFDRAM *mass_fix_Q_dram = new DTFDRAM[qsize * vecdim]();

        DirectQuant<DTFSSD> *QuantSSD = new DirectQuant<DTFSSD>(vecdim, false, true, true);
        QuantSSD->FixDataPoint(mass_global, vecsize, vecsize_dram_total);
        QuantSSD->AddFullDataToFix(mass_global, mass_fix_global, vecsize);
        QuantSSD->AddFullDataToFix(massQ, mass_fix_Q_ssd, qsize);
        delete[] mass_global;

        // printf("compersion\n");
        // for (size_t r = 0; r < vecdim; r++)
        //     printf("%.3f\t", mass_global[241 * vecdim + r]);
        // printf("\n");
        // for (size_t r = 0; r < vecdim; r++)
        //     printf("%d\t", mass_fix_global[241 * vecdim + r]);
        // printf("\n");
        // exit(1);

        printf("Quantization error is %.3f, quantzation number is %d, overflow number is %d\n", 
                QuantSSD->_quant_err, QuantSSD->_quant_nums, QuantSSD->_overflow_nums);
        printf("Quantization for SSD, %d base data and %d queries.\n", vecsize, qsize);

        DirectQuant<DTFDRAM> *QuantDRAM = new DirectQuant<DTFDRAM>(vecdim, true, true, true, vecsize_dram_total, qsize);
        QuantDRAM->FixDataPoint(mass_graph, vecsize_dram_total, vecsize_dram_total);
        QuantDRAM->AddFullDataToFix(mass_graph, mass_fix_graph, vecsize_dram_total, 0);
        QuantDRAM->AddFullDataToFix(massQ, mass_fix_Q_dram, qsize, 1);
        delete[] mass_graph;
        delete[] massQ;

        printf("Quantization error is %.3f, quantzation number is %d, overflow number is %d\n", 
                QuantDRAM->_quant_err, QuantDRAM->_quant_nums, QuantDRAM->_overflow_nums);
        printf("Quantization for DRAM, %d base data and %d queries.\n", vecsize_dram_total, qsize);

        if (mkdir(dir_fix.c_str(), S_IRWXU) != 0) {
            printf("Error, dir %s create failed \n", dir_fix.c_str());
            exit(1);
        }
        WriteBinToArray<DTFSSD>(path_fix_mass_global, mass_fix_global, vecsize, vecdim);
        WriteBinToArray<DTFSSD>(path_fix_mass_Q_ssd, mass_fix_Q_ssd, qsize, vecdim);
        WriteBinToArray<DTFDRAM>(path_fix_mass_graph, mass_fix_graph, vecsize_dram_total, vecdim);
        WriteBinToArray<DTFDRAM>(path_fix_mass_Q_dram, mass_fix_Q_dram, qsize, vecdim);
        WriteBinToArray<uint8_t>(path_fix_flag_graph, QuantDRAM->CoarseTableBase, vecsize_dram_total, QuantDRAM->_coarse_table_len);
        WriteBinToArray<uint8_t>(path_fix_flag_Q_dram, QuantDRAM->CoarseTableQuery, qsize, QuantDRAM->_coarse_table_len);
        delete[] mass_fix_global;
        delete[] mass_fix_Q_ssd;
        delete[] mass_fix_graph;
        delete[] mass_fix_Q_dram;

        printf("file in %s generate is done \n", dir_fix.c_str());
    }

    if (!access(dir_index.c_str(), R_OK|W_OK)){
        printf("Index set %s is existed \n", dir_index.c_str());
        return;
    } else {
        if (mkdir(dir_index.c_str(), S_IRWXU) != 0) {
            printf("Error, dir %s create failed \n", dir_index.c_str());
            exit(1);
        }
        DTset *mass_graph = new DTset[vecsize_dram_total * vecdim]();
        DTFDRAM *mass_fix_graph = new DTFDRAM[vecsize_dram_total * vecdim]();
        uint8_t *flag_fix_graph = new uint8_t[vecsize_dram_total * (unsigned)ceil(vecdim / 8)]();
        LoadBinToArray<DTset>(path_clu_mass_graph, mass_graph, vecsize_dram_total, vecdim);
        LoadBinToArray<DTFDRAM>(path_fix_mass_graph, mass_fix_graph, vecsize_dram_total, vecdim);
        LoadBinToArray<uint8_t>(path_fix_flag_graph, flag_fix_graph, vecsize_dram_total, ceil(vecdim / 8));

        L2Space l2space(vecdim);
        std::vector<HierarchicalNSW<DTres> *> appr_alg(num_banks);
        for (size_t bank_i = 0; bank_i < num_banks; bank_i++){
            string path_index = path_index_prefix + to_string(bank_i) + ".bin";
            appr_alg[bank_i] = new HierarchicalNSW<DTres>(&l2space, vecsize_dram_per_bank, M, efConstruction);

            DTset *massbase = mass_graph + bank_i * vecsize_dram_per_bank * vecdim;
            DTFDRAM *massfix = mass_fix_graph + bank_i * vecsize_dram_per_bank * vecdim;
            uint8_t *flagfix = flag_fix_graph + bank_i * vecsize_dram_per_bank * (unsigned)ceil(vecdim / 8);
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

            printf("Replace feature for index: %u \n", bank_i);
            appr_alg[bank_i]->ReplaceFeature(massfix, flagfix, vecsize_dram_per_bank);

            if (isSave)
                appr_alg[bank_i]->saveIndex(path_index);
        }
        delete[] mass_graph;
        delete[] mass_fix_graph;
        delete[] flag_fix_graph;

        printf("Build index in dir %s is succeed \n", dir_index.c_str());
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

    string path_fix_mass_global = dir_fix + "/mass_global.bin";
    string path_fix_mass_Q_ssd = dir_fix + "/mass_Q_ssd.bin";
    string path_fix_mass_Q_dram = dir_fix + "/mass_Q_dram.bin";
    string path_fix_flag_Q_dram = dir_fix + "/flag_Q_dram.bin";
    string path_clu_global_label_to_id = dir_clu + "/global_label_to_id.bin";

    string path_index_prefix = dir_index + "/index_bank_";

    if (access(dir_index.c_str(), R_OK|W_OK)){
        printf("Error, index %s is unexisted \n", dir_index.c_str());
        exit(1);
    } else {
        size_t flag_len = ceil(vecdim / 8);
        unsigned *massQA = new unsigned[qsize * gt_maxnum];
        // DTres *massQA_dist = new DTres[qsize * gt_maxnum];
        DTset *massQ = new DTset[qsize * vecdim];
        // DTFSSD *mass_fix_global = new DTFSSD[vecsize * vecdim]();
        DTFSSD *mass_fix_Q_ssd = new DTFSSD[qsize * vecdim]();
        DTFDRAM *mass_fix_Q_dram = new DTFDRAM[qsize * vecdim]();
        uint8_t *flag_fix_Q_dram = new uint8_t[qsize * flag_len]();
        unsigned *global_label_to_id = new unsigned[vecsize]();

        cout << "Loading GT:\n";
        // LoadBinToArray<unsigned>(index_string["gt_id"], massQA, qsize, gt_maxnum);
        // LoadBinToArray<DTres>(index_string["gt_dist"], massQA_dist, qsize, gt_maxnum);
        ifstream inputGT(path_gt.c_str(), ios::binary);
        for (int i = 0; i < qsize; i++) {
            int t;
            inputGT.read((char *) &t, 4);
            inputGT.read((char *) (massQA + gt_maxnum * i), gt_maxnum * 4);
            if (t != gt_maxnum) {
                cout << "err";
                return;
            }
        }
        inputGT.close();
        printf("Load GT from %s done \n", path_gt.c_str());
        
        cout << "Loading queries:\n";
        LoadBinToArray<DTFSSD>(path_fix_mass_Q_ssd, mass_fix_Q_ssd, qsize, vecdim);
        LoadBinToArray<DTFDRAM>(path_fix_mass_Q_dram, mass_fix_Q_dram, qsize, vecdim);
        LoadBinToArray<uint8_t>(path_fix_flag_Q_dram, flag_fix_Q_dram, qsize, flag_len);
        LoadBinToArray<unsigned>(path_clu_global_label_to_id, global_label_to_id, vecsize_dram_total, num_perspnode);
        printf("Load fixed queries and dram flag done \n");

        L2SpaceIntFlag l2if(vecdim);
        std::vector<HierarchicalNSW<FCP32> *> appr_alg(num_banks);
        for (size_t bank_i = 0; bank_i < num_banks; bank_i++){
            string path_index = path_index_prefix + to_string(bank_i) + ".bin";
            appr_alg[bank_i] = new HierarchicalNSW<FCP32>(&l2if, path_index, false);
        }
    
        vector<std::priority_queue<std::pair<FCP64, labeltype >>> answers;
        cout << "Parsing gt:\n";
        get_gt(massQA, qsize, gt_maxnum, vecdim, answers, k);

        cout << "Comput recall: \n";
        size_t correct = 0;
        size_t total = 0;
        size_t len_per_query_comb = vecdim * sizeof(DTFDRAM) + flag_len;

        size_t efs_bank = 150;
        size_t efs_dram = num_banks * efs_bank * EFS_PROP;
        
        DTFSSD *mass_brute_ssd = new DTFSSD[efs_dram * num_perspnode * vecdim]();
        unsigned *id_brute_ssd = new unsigned[efs_dram * num_perspnode]();

        for (size_t q_i = 0; q_i < qsize; q_i++){
            // search stage 1: in DRAM
            char *query_comb = new char[len_per_query_comb]();
            memcpy(query_comb, mass_fix_Q_dram + q_i * vecdim, vecdim * sizeof(DTFDRAM));
            memcpy(query_comb + vecdim * sizeof(DTFDRAM), flag_fix_Q_dram + q_i * flag_len, flag_len);
            FCP32 thr_global = std::numeric_limits<FCP32>::max();
            // space_l2
            char *query_repeat = new char[num_banks * len_per_query_comb]();
            std::vector<std::priority_queue<std::pair<FCP32, labeltype>>> result_bank(num_banks);

            std::priority_queue<std::pair<FCP32, labeltype>> result_dram;

            omp_set_num_threads(num_banks);
#pragma omp parallel for
            for (size_t bank_i = 0; bank_i < num_banks; bank_i++){
                // appr_alg[bank_i]->setThr(&thr_global);
                appr_alg[bank_i]->setEf(efs_bank);

                char *query_c = query_repeat + bank_i * len_per_query_comb;
                memcpy(query_c, query_comb, len_per_query_comb);
                // todo 
                result_bank[bank_i] = appr_alg[bank_i]->searchKnn(query_c, efs_bank);
#pragma omp critical
                {
                    // merge result
                    while (!result_bank[bank_i].empty()){
                        result_dram.emplace(std::make_pair(result_bank[bank_i].top().first, 
                                                (bank_i * vecsize_dram_per_bank + result_bank[bank_i].top().second)));
                        result_bank[bank_i].pop();
                    }
                    // 
                    while(result_dram.size() > efs_dram){
                        result_dram.pop();
                    }
                }
            }
            // DTFSSD *fsb = new DTFSSD[vecsize * vecdim]();
            // LoadBinToArray<DTFSSD>(path_fix_mass_global, fsb, vecsize, vecdim);
            // for (size_t i = 0; i < vecsize; i++){
            //     printf("%u, id %u: ", i, global_label_to_id[i]);
            //     for (size_t j = 0; j < vecdim; j++)
            //         printf("%d\t", fsb[i * vecdim + j]);
            //     printf("\n");
            // }
            // exit(1);

            // printf("----\n");
            // for (size_t r = 0; r < gt_maxnum; r++){
            //     printf("%u\t", massQA[q_i * gt_maxnum + r]);
            //     // vector<labeltype> innode;
            //     // appr_alg[0]->getIndegreeByExternal(massQA[q_i * gt_maxnum + r], innode);
            //     // printf("%u 's indegree: %u \n", massQA[q_i * gt_maxnum + r], innode.size());
            // }
            // printf("\n");
            // for (size_t r = 0; r < gt_maxnum; r++)
            //     printf("%.4f\t", massQA_dist[q_i * gt_maxnum + r]);
            // printf("\n\n");

            // while(!result_dram.empty()){
            //     if (result_dram.size() <= k)
            //         printf("%u: %d\n", global_label_to_id[result_dram.top().second], result_dram.top().first);
            //     result_dram.pop();
            // }
            // exit(1);

            // search in SSD
            size_t efs_real = std::min(efs_dram, result_dram.size());
            memset(mass_brute_ssd, 0, efs_dram * num_perspnode * vecdim * sizeof(DTFSSD));
            memset(id_brute_ssd, 0, efs_dram * num_perspnode * sizeof(unsigned));

            ifstream inputBS(path_fix_mass_global.c_str(), ios::binary);
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
                memcpy(id_brute_ssd + i * num_perspnode, global_label_to_id + pos_r * num_perspnode, num_perspnode * sizeof(unsigned));
            }
            inputBS.close();

            L2SpaceSSD l2ssd(vecdim);
            BruteforceSearch<FCP64>* brute_alg = new BruteforceSearch<FCP64>(&l2ssd, (size_t)(efs_real * num_perspnode));
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

            std::priority_queue<std::pair<FCP64, labeltype >> rs = brute_alg->searchKnn(mass_fix_Q_ssd + q_i * vecdim, k);
            
            std::priority_queue<std::pair<FCP64, labeltype >> gt(answers[q_i]);
            unordered_set<labeltype> g;
            total += gt.size();

            while (gt.size()) {
                g.insert(gt.top().second);
                gt.pop();
            }

            while (rs.size()) {
                if (g.find(rs.top().second) != g.end()) {
                    correct++;
                }
                rs.pop();
            }

        }

        float recall = 1.0f * correct / total;
        cout << efs_bank << "\t" << recall << "\n";

        printf("Search index %s is succeed \n", dir_index.c_str());
    }
}

// need to support:
// 

void hnsw_impl(bool is_build, const string &using_dataset){
    string prefix = "/home/ljun/anns/hnsw_nics/graphindex/";
    string label = "nmp/";
    // support dataset: sift, gist, deep, glove, crawl

	size_t subset_size_milllions = 1;
	size_t efConstruction = 40;
	size_t M = 16;
    size_t k = 10;
	
    size_t vecsize = subset_size_milllions * 1000000;
    size_t qsize, vecdim, gt_maxnum;
    string path_index, path_gt, path_q, path_data;
    

    size_t num_banks = NUM_BANKS;
    size_t num_perspnode = NUM_PERSPNODE;

    std::map<string, size_t> index_parameter;
    index_parameter["efConstruction"] = efConstruction;
    index_parameter["M"] = M;
    index_parameter["k"] = k;
    index_parameter["vecsize"] = vecsize;
    index_parameter["num_banks"] = num_banks;
    index_parameter["num_perspnode"] = num_perspnode;
    std::map<string, string> index_string;

    /*
    file structure
    --nmp
       |--deep10m_bk8pspn10
            |--clu_data
            |--fix_data
            |--index
    */
    string dir_this = prefix + label + using_dataset + to_string(subset_size_milllions) + 
                        "m_bk" + to_string(num_banks) + "spn" + to_string(num_perspnode);
    if (access(dir_this.c_str(), R_OK|W_OK)){
        if (mkdir(dir_this.c_str(), S_IRWXU) != 0) {
            printf("Error, dir %s create failed \n", dir_this.c_str());
            exit(1);
        }
    }
    
    index_string["dir_clu"] = dir_this + "/clu_data";
    index_string["dir_fix"] = dir_this + "/fix_data";
    index_string["dir_index"] = dir_this + "/ef" + to_string(efConstruction) + "m" + to_string(M);

    CheckDataset(using_dataset, index_parameter, index_string, subset_size_milllions, vecsize, qsize, vecdim, gt_maxnum,
                    path_q, path_data, path_gt);
    
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
