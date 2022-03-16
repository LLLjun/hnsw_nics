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
    // omp_set_num_threads(80);
// #pragma omp parallel for
    for (int i = 0; i < qsize; i++) {
#endif
        std::priority_queue<std::pair<DTres, labeltype >> result = appr_alg.searchKnn(massQ + vecdim * i, k);
        std::priority_queue<std::pair<DTres, labeltype >> gt(answers[i]);
        unordered_set<labeltype> g;
        
        while (gt.size()) {
            g.insert(gt.top().second);
            gt.pop();
        }

// #pragma omp critical
//         {
        total += g.size();
        while (result.size()) {
            if (g.find(result.top().second) != g.end()) {
                correct++;
            } else {
            }
            result.pop();
        }
        // }
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

    cout << "efs\t" << "R@" << k << "\t" << "NDC_avg\t" << "qps\n";
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
        StopW stopw = StopW();

        float recall = test_approx(massQ, qsize, appr_alg, vecdim, answers, k);
        float time_us_per_query = stopw.getElapsedTimeus() / qsize;
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
        // cout << ef << "\t" << recall << "\t" << NDC_avg << "\t" << time_us_per_query << "\n";
        cout << ef << "\t" << recall << "\t" << NDC_avg << "\t" << (1e6 / time_us_per_query)
        << "\n";
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

template<typename DTset, typename DTval, typename DTres>
void build_index(const string &dataname, map<string, size_t> &index_parameter, map<string, string> &index_string, bool isSave = true){
    //
    size_t efConstruction = index_parameter["efConstruction"];
    size_t M = index_parameter["M"];
    size_t vecsize = index_parameter["vecsize"];
    size_t vecdim = index_parameter["vecdim"];
    size_t qsize = index_parameter["qsize"];

    string path_data = index_string["path_data"];
    string format = index_string["format"];
    string index = index_string["index"];

    if (exists_test(index)){
        printf("Index %s is existed \n", index.c_str());
        return;
    } else {

        DTset *massB = new DTset[vecsize * vecdim]();
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

        L2Space l2space(vecdim);
        HierarchicalNSW<DTres> *appr_alg = new HierarchicalNSW<DTres>(&l2space, vecsize, M, efConstruction);
#if PLATG
        unsigned center_id = compArrayCenter<DTset>(massB, vecsize, vecdim);
        appr_alg->addPoint((void *) (massB + center_id * vecdim), (size_t) center_id);
#else
        appr_alg->addPoint((void *) (massB), (size_t) 0);
#endif
        cout << "Building index:\n";
        int j1 = 0;
        StopW stopw = StopW();
        StopW stopw_full = StopW();
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
    }
}

template<typename DTset, typename DTval, typename DTres>
void search_index(const string &dataname, map<string, size_t> &index_parameter, map<string, string> &index_string){
    //
    size_t k = index_parameter["k"];
    size_t vecsize = index_parameter["vecsize"];
    size_t qsize = index_parameter["qsize"];
    size_t vecdim = index_parameter["vecdim"];
    size_t gt_maxnum = index_parameter["gt_maxnum"];

    string path_q = index_string["path_q"];
    string format = index_string["format"];
    string index = index_string["index"];
    string path_gt = index_string["path_gt"];

    if (!exists_test(index)){
        printf("Error, index %s is unexisted \n", index.c_str());
        exit(1);
    } else {

        unsigned *massQA = new unsigned[qsize * gt_maxnum];
        DTset *massQ = new DTset[qsize * vecdim];

        cout << "Loading GT:\n";
        LoadBinToArray<unsigned>(path_gt, massQA, qsize, gt_maxnum);
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

        L2Space l2space(vecdim);
        HierarchicalNSW<DTres> *appr_alg = new HierarchicalNSW<DTres>(&l2space, index, false);

        vector<std::priority_queue<std::pair<DTres, labeltype >>> answers;
        cout << "Parsing gt:\n";
        get_gt(massQA, qsize, gt_maxnum, vecdim, answers, k);

#if MEMTRACE
        appr_alg->initMem();
#endif

        cout << "Comput recall: \n";
        test_vs_recall(massQ, qsize, *appr_alg, vecdim, answers, k);

#if MEMTRACE
        string file_mem_trace = "/home/usr-xkIJigVq/nmp/hnsw_nics/output/mem/trace.txt";
        appr_alg->main_mem->write_file(file_mem_trace, appr_alg->main_mem->count_trace('a'));
#endif

        printf("Search index %s is succeed \n", index.c_str());
    }
}

void hnsw_impl(bool is_build, const string &using_dataset){
    string path_project = "/home/usr-xkIJigVq/nmp/hnsw_nics";
#if PLATG
    string label = "plat/";
#else
    string label = "base/";
#endif
    string path_graphindex = path_project + "/graphindex/" + label;

    string pre_index = path_graphindex + using_dataset;
    if (access(pre_index.c_str(), R_OK|W_OK)){
        if (mkdir(pre_index.c_str(), S_IRWXU) != 0) {
            printf("Error, dir %s create failed \n", pre_index.c_str());
            exit(1);
        }
    }

	size_t subset_size_milllions = 1;
	size_t efConstruction = 200;
	size_t M = 20;
    size_t k = 10;
#if AKNNG
    // subset_size_milllions = 10;
    k = 100;
    if (subset_size_milllions == 10){
        efConstruction = 400;
        M = 30;
    }
#endif

    size_t vecsize = subset_size_milllions * 1000000;
    size_t qsize, vecdim, gt_maxnum;
    string path_index, path_gt, path_q, path_data;

    std::map<string, size_t> index_parameter;
    index_parameter["subset_size_milllions"] = subset_size_milllions;
    index_parameter["efConstruction"] = efConstruction;
    index_parameter["M"] = M;
    index_parameter["k"] = k;
    index_parameter["vecsize"] = vecsize;

    std::map<string, string> index_string;
    index_string["format"] = "float";

    string hnsw_index = pre_index + "/" + using_dataset + to_string(subset_size_milllions) +
                        "m_ef" + to_string(efConstruction) + "m" + to_string(M) + ".bin";
    index_string["index"] = hnsw_index;
    CheckDataset(using_dataset, index_parameter, index_string);

    L2Space l2space(vecdim);

    if (is_build){
        build_index<DTSET, DTVAL, DTRES>(using_dataset, index_parameter, index_string);
    } else{
        search_index<DTSET, DTVAL, DTRES>(using_dataset, index_parameter, index_string);
    }
    return;
}
