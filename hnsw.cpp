#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include "hnswlib/hnswlib.h"
#include <unordered_set>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/types.h>

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
#if MEMTRACE
    {   int i = 100;
#else

//     omp_set_num_threads(3);
// #pragma omp parallel for
    for (int i = 0; i < qsize; i++) {
#endif

#if RANKMAP
        std::priority_queue<std::pair<DTres, labeltype >> result = appr_alg.searchParaRank(massQ + vecdim * i, k);
#else
        std::priority_queue<std::pair<DTres, labeltype >> result = appr_alg.searchKnn(massQ + vecdim * i, k);
#endif
        std::priority_queue<std::pair<DTres, labeltype >> gt(answers[i]);
        unordered_set<labeltype> g;

        while (gt.size()) {
            g.insert(gt.top().second);
            gt.pop();
        }

// #pragma omp critical
        {
            total += g.size();
            while (result.size()) {
                if (g.find(result.top().second) != g.end()) {
                    correct++;
                } else {
                }
                result.pop();
            }
        }
    }
    return 1.0f * correct / total;
}

template<typename DTval, typename DTres>
static void
test_vs_recall(DTval *massQ, size_t qsize, HierarchicalNSW<DTres> &appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<DTres, labeltype >>> &answers, size_t k) {
    vector<size_t> efs;// = { 10,10,10,10,10 };
#if MEMTRACE
    efs.push_back(80);
#elif AKNNG
    for (int i = 200; i <= 700; i += 100)
        efs.push_back(i);
#else
    for (int i = 10; i <= 150; i += 10)
        efs.push_back(i);
#endif

    cout << "efs\t" << "R@" << k << "\t" << "time_us" << "\t";
#if RANKMAP
    if (appr_alg.stats != nullptr) {
        cout << "rank_us\t" << "hlc_us\t";
        cout << "NDC_avg\t" << "NDC_max\t" << "n_hops\t";
    }
#endif
    cout << endl;

    for (size_t ef : efs) {
        appr_alg.setEf(ef);
        appr_alg.metric_hops = 0;
        appr_alg.metric_hops_L = 0;
        appr_alg.metric_distance_computations = 0;
        // appr_alg.hits_pre_comput = 0;
#if PROFILE
        appr_alg.time_PDC = 0;
        appr_alg.time_sort = 0;
#endif

#if RANKMAP
        if (appr_alg.stats != nullptr) {
            appr_alg.stats->n_max_NDC = 0;
            appr_alg.stats->hlc_us = 0;
            appr_alg.stats->rank_us = 0;
            appr_alg.stats->n_hops = 0;
        }
#endif

        Timer stopw = Timer();

        float recall = test_approx(massQ, qsize, appr_alg, vecdim, answers, k);
        float time_us_per_query = (double) stopw.getElapsedTimeus() / qsize;
        float avg_hop_0 = 1.0f * appr_alg.metric_hops / qsize;
        float avg_hop_L = 1.0f * appr_alg.metric_hops_L / qsize;
        float NDC_avg = 1.0f * appr_alg.metric_distance_computations / qsize;
        // float hits = (float) appr_alg.hits_pre_comput / (appr_alg.metric_hops - 1);

#if PROFILE
        float TDC = appr_alg.time_PDC / qsize;
        float Tsort = appr_alg.time_sort / qsize;
        cout << ef << "\t" << recall << "\t" << NDC_avg << "\t" << time_us_per_query << "\t" <<
                TDC << "\t" << Tsort << "\n";
#else
        cout << ef << "\t" << recall << "\t" << time_us_per_query << "\t";
#if RANKMAP
        if (appr_alg.stats != nullptr) {
            cout << appr_alg.stats->rank_us / qsize << "\t";
            cout << appr_alg.stats->hlc_us / qsize << "\t";
            cout << NDC_avg << "\t";
            cout << appr_alg.stats->n_max_NDC / qsize << "\t";
            cout << appr_alg.stats->n_hops / qsize << "\t";
        }
#endif
        cout << endl;
#endif
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


template<typename DTset, typename DTres>
void search_index(const string &dataname, map<string, size_t> &index_parameter, map<string, string> &index_string, bool isTrans){
    //
    size_t vecsize = index_parameter["vecsize"];
    size_t vecdim = index_parameter["vecdim"];
    size_t qsize = index_parameter["qsize"];
    size_t gt_maxnum = index_parameter["gt_maxnum"];
    size_t k = gt_maxnum;

    string path_data = index_string["path_data"];
    string path_q = index_string["path_q"];
    string path_gt = index_string["path_gt"];

    cout << "path_data: " << path_data.c_str() << "\n"
         << "path_q: " << path_q.c_str() << "\n"
         << "path_gt: " << path_gt.c_str() << "\n"
         << "trans?: " << isTrans << "\n"
         << "vecsize: " << vecsize << "\n"
         << "vecdim: " << vecdim << "\n";

    {
        DTset *massQA = new DTset[vecsize * vecdim];
        LoadBinToArrayIghead<DTset>(path_data, massQA, vecsize, vecdim);

        DTset *massQA_test = new DTset[vecsize * vecdim];
        string path_gt_test = path_data + "_rewrite";
        LoadBinToArray<DTset>(path_gt_test, massQA_test, vecsize, vecdim);

        for (int i = 0; i < vecsize; i++){
            for (int j = 0; j < vecdim; j++){
                if (massQA[i*vecdim+j] != massQA_test[i*vecdim+j]){
                    printf("%d, %d: %.3f, %.3f\n", i, j, massQA[i*vecdim+j], massQA_test[i*vecdim+j]);   
                }
                // printf("%d, %d: %.3f, %.3f\n", i, j, massQA[i*vecdim+j], massQA_test[i*vecdim+j]);
            }
            // exit(1);
        }
        exit(1);
    }

    DTset *massB = new DTset[vecsize * vecdim]();
    cout << "Loading base data:\n";
    if (isTrans){
        LoadBinToArrayIghead<DTset>(path_data, massB, vecsize, vecdim);
        string path_test = path_data + "_rewrite";
        WriteBinToArray<DTset>(path_test, massB, vecsize, vecdim);
        // exit(1);
    } else
        LoadBinToArray<DTset>(path_data, massB, vecsize, vecdim);
    
#if FMTINT
    L2SpaceI l2space(vecdim);
#else
    L2Space l2space(vecdim);
#endif

    cout << "Building index:\n";
    BruteforceSearch<DTres>* brute_alg = new BruteforceSearch<DTres>(&l2space, vecsize);
#pragma omp parallel for
    for (size_t i = 0; i < vecsize; i++){
        brute_alg->addPoint((void *) (massB + i * vecdim), i);
    }
    printf("Build index is succeed \n");
    delete[] massB;

    unsigned *massQA = new unsigned[qsize * gt_maxnum];
    DTset *massQ = new DTset[qsize * vecdim];

    cout << "Loading queries:\n";
    LoadBinToArray<DTset>(path_q, massQ, qsize, vecdim);

    printf("Begin to search groundtruth\n");
    Timer ts = Timer();
    int ti = 0;
#pragma omp parallel for
    for (size_t i = 0; i < qsize; i++){
        std::priority_queue<std::pair<DTres, labeltype>> rs = brute_alg->searchKnn(massQ + i * vecdim, k);
        size_t kk = k;
        while (!rs.empty()){
            massQA[i * gt_maxnum + kk - 1] = (unsigned) rs.top().second;
            rs.pop();
            kk--;
        }
        if (kk != 0){
            printf("Error, expect nums: %u \n", k);
            exit(1);
        }
#pragma omp critical
        {
            ti++;
            if ((ti * 10) % qsize == 0){
                int tt = (ti * 10) / qsize;
                printf("%d / 10, %.1f s\n", tt, ts.getElapsedTimeus() * 1e-6);
            }
        }
    }

    cout << "Writing GT:\n";
    WriteBinToArray<unsigned>(path_gt, massQA, qsize, gt_maxnum);
    printf("Write GT to %s done \n", path_gt.c_str());

    delete[] massQ;
    delete[] massQA;
}

// 修改 base vector 的数量（可选），以及生成真实值
void hnsw_impl(const string &using_dataset, size_t sizeVectorM, string isTrans){

	size_t subset_size_milllions = sizeVectorM;
    size_t vecsize = subset_size_milllions * 1000000;

    std::map<string, size_t> index_parameter;
    index_parameter["subset_size_milllions"] = subset_size_milllions;
    index_parameter["vecsize"] = vecsize;

    std::map<string, string> index_string;

    CheckDataset(using_dataset, index_parameter, index_string);

    search_index<DTSET, DTRES>(using_dataset, index_parameter, index_string, (isTrans == "trans"));

    return;
}
