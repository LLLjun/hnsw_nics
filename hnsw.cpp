#include <iostream>
#include <fstream>
#include <queue>
#include <map>
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
    for (int i = 40; i < 100; i += 10) {
        efs.push_back(i);
    }
    for (int i = 100; i < 500; i += 40) {
        efs.push_back(i);
    }
    for (int i = 500; i < 1500; i += 200) {
        efs.push_back(i);
    }
    // for (int i = 1500; i < 10500; i += 1000) {
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

template<typename DTset, typename DTval, typename DTres>
void build_index(const string &dataname,  SpaceInterface<DTres> &s, 
                map<string, size_t> &index_parameter, map<string, string> &index_string, bool isSave = true){           
    // 
    string index = index_string["hnsw_index"];
    size_t efConstruction = index_parameter["efConstruction"];
    size_t M = index_parameter["M"];
    size_t vecsize = index_parameter["vecsize"];
    size_t vecdim = index_parameter["vecdim"];
    string path_data = index_string["path_data"];
    string graph_type = "base";

    if (exists_test(index)){
        printf("Index %s is existed \n", index.c_str());
        return;
    } else {

        DTset *massB = new DTset[vecsize * vecdim]();

        cout << "Loading base data:\n";
        LoadBinToArray<DTset>(path_data, massB, vecsize, vecdim);

        HierarchicalNSW<DTres> *appr_alg = new HierarchicalNSW<DTres>(&s, vecsize, M, efConstruction);
        // appr_alg->testSortMultiadd();
        // exit(0);
        
        appr_alg->graph_type = graph_type;
        appr_alg->hit_miss = 0;
        appr_alg->hit_total = 0;
#if PLATG
        unsigned center_id = compArrayCenter<DTset>(massB, vecsize, vecdim);
        appr_alg->addPoint((void *) (massB + center_id * vecdim), (size_t) center_id);
#else
        appr_alg->addPoint((void *) (massB), (size_t) 0);
#endif
        cout << "Building index:\n";
        int j1 = 0;
        clk_get stopw = clk_get();
        clk_get stopw_full = clk_get();
        size_t report_every = vecsize / 10;
#pragma omp parallel for
        for (size_t i = 1; i < vecsize; i++) {
#pragma omp critical
            {
                j1++;
                if (j1 % report_every == 0) {
                    cout << j1 / (0.01 * vecsize) << " %, "
                         << report_every / (1000.0 * stopw.getElapsedTimes()) << " kips " << " Mem: "
                         << getCurrentRSS() / 1000000 << " Mb \n";
                    stopw.reset();
                }
            }
#if PLATG
            size_t ic;
            if (i <= center_id)
                ic = i - 1;
            else
                ic = i;
            appr_alg->addPoint((void *) (massB + ic * vecdim), ic);
#else
            appr_alg->addPoint((void *) (massB + i * vecdim), i);
#endif
        }
        cout << "Build time:" << stopw_full.getElapsedTimes() << "  seconds\n";
        delete[] massB;
        if (isSave)
            appr_alg->saveIndex(index);

        printf("Build index %s is succeed \n", index.c_str());

        // get average neighbor distance
        // float dist_a = appr_alg->compAverageNeighDist();
        // printf("average neighbor distance: %.3f \n", dist_a);
        // float dist_b = appr_alg->compNeighborDistByDegree();
        // printf("average neighbor distance (degree = 1): %.3f \n", dist_b);

        // 
        // appr_alg->getDegreeRelation();

        // get degree info
        vector<size_t> dstb_in;
        vector<size_t> dstb_out;
        appr_alg->getDegreeDistri(dstb_in, dstb_out);

        // // get her hit
        // size_t hit_miss = appr_alg->hit_miss;
        // size_t hit_total = appr_alg->hit_total;
        // printf("miss: %u, total: %u, prop: %.3f \n", hit_miss, hit_total, (float)hit_miss / hit_total);
    }
}

template<typename DTset, typename DTval, typename DTres>
void search_index(const string &dataname, SpaceInterface<DTres> &s, 
                map<string, size_t> &index_parameter, map<string, string> &index_string){
    // 
    string index = index_string["hnsw_index"];
    string path_q = index_string["path_q"];
    string path_gt = index_string["path_gt"];

    size_t vecdim = index_parameter["vecdim"];
    size_t k = index_parameter["k"];
    size_t qsize = index_parameter["qsize"];
    size_t gt_maxnum = index_parameter["gt_maxnum"];

    if (!exists_test(index)){
        printf("Error, index %s is unexisted \n", index.c_str());
        exit(1);
    } else {

        unsigned *massQA = new unsigned[qsize * gt_maxnum];
        DTset *massQ = new DTset[qsize * vecdim];

        cout << "Loading GT:\n";
        LoadBinToArray<unsigned>(path_gt, massQA, qsize, gt_maxnum);
        printf("Load GT from %s done \n", path_gt.c_str());
        
        cout << "Loading queries:\n";
        LoadBinToArray<DTset>(path_q, massQ, qsize, vecdim);
        printf("Load queries from %s done \n", path_q.c_str());

        HierarchicalNSW<DTres> *appr_alg = new HierarchicalNSW<DTres>(&s, index, false);
    
        vector<std::priority_queue<std::pair<DTres, labeltype >>> answers;
        cout << "Parsing gt:\n";
        get_gt(massQA, qsize, gt_maxnum, vecdim, answers, k);

        cout << "Comput recall: \n";
        test_vs_recall(massQ, qsize, *appr_alg, vecdim, answers, k);

        // // get degree info
        // vector<size_t> dstb_in;
        // vector<size_t> dstb_out;
        // appr_alg->getDegreeDistri(dstb_in, dstb_out);

        printf("Search index %s is succeed \n", index.c_str());
        delete[] massQA;
        delete[] massQ;
    }
}

void hnsw_impl(bool is_build, const string &using_dataset, string &graph_type){
    string root_index = "/home/usr-xkIJigVq/vldb/hnsw_nics/graphindex/";
    string root_output = "/home/usr-xkIJigVq/vldb/hnsw_nics/output/";

    string label = "hnsw/";

    // support dataset: sift, gist, deep, glove, crawl

    string pre_index = root_index + label + using_dataset;
    if (access(pre_index.c_str(), R_OK|W_OK)){
        if (mkdir(pre_index.c_str(), S_IRWXU) != 0) {
            printf("Error, dir %s create failed \n", pre_index.c_str());
            exit(1);
        }
    }
    string pre_output = root_output + label + using_dataset;
    if (access(pre_output.c_str(), R_OK|W_OK)){
        if (mkdir(pre_output.c_str(), S_IRWXU) != 0) {
            printf("Error, dir %s create failed \n", pre_output.c_str());
            exit(1);
        }
    }

	size_t subset_size_milllions = 1;
	size_t efConstruction = 40;
	size_t M = 16;
    size_t k = 10;
	
    size_t vecsize = subset_size_milllions * 1000000;
    size_t qsize, vecdim, gt_maxnum;
    string path_index, path_gt, path_q, path_data;
    
#if EXI
    string hnsw_index = pre_index + "/" + using_dataset + to_string(subset_size_milllions) + 
                        "m_ef" + to_string(efConstruction) + "m" + to_string(M) + "_" + graph_type + "_exi.bin";
#else
    string hnsw_index = pre_index + "/" + using_dataset + to_string(subset_size_milllions) + 
                        "m_ef" + to_string(efConstruction) + "m" + to_string(M) + "_" + graph_type + ".bin";
#endif

    map<string, size_t> index_parameter;
    index_parameter["subset_size_milllions"] = subset_size_milllions;
    index_parameter["efConstruction"] = efConstruction;
    index_parameter["M"] = M;
    index_parameter["k"] = k;
    index_parameter["vecsize"] = vecsize;

    map<string, string> index_string;
    index_string["using_dataset"] = using_dataset;
    index_string["hnsw_index"] = hnsw_index;
    index_string["log_output"] = pre_output + "/" + using_dataset + to_string(subset_size_milllions) + "m.log";

    CheckDataset(using_dataset, index_parameter, index_string);
    
    L2Space l2space(index_parameter["vecdim"]);

    printf("%s\n", index_string["path_data"].c_str());

    if (is_build){
        build_index<DTSET, DTVAL, DTRES>(using_dataset, l2space, index_parameter, index_string);
    } else{
        search_index<DTSET, DTVAL, DTRES>(using_dataset, l2space, index_parameter, index_string);
    }
    return;
}
