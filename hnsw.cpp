#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <queue>
#include <chrono>
#include <memory>
#include "hnswlib/hnswlib.h"
#include <string>
#include <unordered_set>
#include <vector>

using namespace std;
using namespace hnswlib;

float comput_recall(vector<vector<unsigned>>& result,
                    vector<vector<unsigned>>& massQA, size_t qsize, size_t k){
    size_t correct = 0;
    size_t total = qsize * k;

    for (int qi = 0; qi < qsize; qi++){
        unordered_set<unsigned> g;
        for (int i = 0; i < k; i++)
            g.insert(result[qi][i]);

        for (int i = 0; i < k; i++) {
            if (g.find(massQA[qi][i]) != g.end())
                correct++;
        }
    }
    return (1.0 * correct / total);
}

template<typename DTres, typename DTset>
static void
test_vs_recall(HierarchicalNSW<DTres, DTset>& appr_alg,
                DTset *massQ, vector<vector<unsigned>>& massQA,
                map<string, size_t> &MapParameter,
                map<size_t, vector<float>>& mapResult) {
    size_t qsize = MapParameter["qsize"];
    size_t vecdim = MapParameter["vecdim"];
    size_t k = MapParameter["k"];

    vector<size_t> efs;
    for (int i = 50; i < 100; i += 5)
        efs.push_back(i);
    for (int i = 100; i <= 200; i += 10)
        efs.push_back(i);

    int column_map = 3;

    cout << "efs\t" << "R@" << k << "\t" << "time_us\t";
#if (RANKMAP && STAT)
        cout << "rank_us\t" << "sort_us\t" << "hlc_us\t" << "visited_us\t";
        cout << "NDC_max\t" << "NDC_total\t" << "n_hops\t";
        column_map = 9;
#else
    cout << "n_hop_L\t" << "n_hop_0\t" << "NDC\t";
#endif
    cout << endl;

    vector<float> result_ef(column_map);

    for (size_t ef : efs) {
        appr_alg.setEf(ef);

#if (RANKMAP && STAT)
        appr_alg.stats->Reset();
#else
        appr_alg.metric_hops = 0;
        appr_alg.metric_hops_L = 0;
        appr_alg.metric_distance_computations = 0;
#endif

        vector<vector<unsigned>> result(qsize);
        for (vector<unsigned>& r: result)
            r.resize(k, 0);

        Timer stopw = Timer();
#if THREAD
#pragma omp parallel for
#endif
        for (int qi = 0; qi < qsize; qi++) {
#if RANKMAP
            priority_queue<pair<DTres, labeltype>> res = appr_alg.searchParaRank(massQ + vecdim * qi, k);
#else
            priority_queue<pair<DTres, labeltype>> res = appr_alg.searchKnn(massQ + vecdim * qi, k);
#endif

#if THREAD
#pragma omp critical
#endif
            {
                int i = 0;
                while (!res.empty()){
                    result[qi][i] = (unsigned) res.top().second;
                    res.pop();
                    i++;
                }
            }
        }
        float time_us_per_query = stopw.getElapsedTimeus() / qsize;
        float recall = comput_recall(result, massQA, qsize, k);

#if (RANKMAP && STAT)
        time_us_per_query = appr_alg.stats->hw_us / qsize;
#endif

        cout << ef << "\t" << recall << "\t" << time_us_per_query << "\t";
        result_ef[0] = recall;
        result_ef[1] = time_us_per_query;
        result_ef[2] = 1.0 * appr_alg.metric_distance_computations / qsize;
#if (RANKMAP && STAT)
            cout << appr_alg.stats->all_rank_us / qsize << "\t";
            cout << appr_alg.stats->all_sort_us / qsize << "\t";
            cout << appr_alg.stats->all_hlc_us / qsize << "\t";
            cout << appr_alg.stats->all_visited_us / qsize << "\t";
            result_ef[2] = appr_alg.stats->all_rank_us / qsize;
            result_ef[3] = appr_alg.stats->all_sort_us / qsize;
            result_ef[4] = appr_alg.stats->all_hlc_us / qsize;
            result_ef[5] = appr_alg.stats->all_visited_us / qsize;

            cout << (1.0 * appr_alg.stats->all_n_DC_max / qsize) << "\t";
            cout << (1.0 * appr_alg.stats->all_n_DC_total / qsize) << "\t";
            cout << (1.0 * appr_alg.stats->all_n_hops / qsize) << "\t";
            result_ef[6] = 1.0 * appr_alg.stats->all_n_DC_max / qsize;
            result_ef[7] = 1.0 * appr_alg.stats->all_n_DC_total / qsize;
            result_ef[8] = 1.0 * appr_alg.stats->all_n_hops / qsize;
#else
        cout << (1.0 * appr_alg.metric_hops_L / qsize) << "\t";
        cout << (1.0 * appr_alg.metric_hops / qsize) << "\t";
        cout << (1.0 * appr_alg.metric_distance_computations / qsize) << "\t";
#endif
        cout << endl;

        mapResult.insert(pair<size_t, vector<float>>(ef, result_ef));

        if (recall > 1.0) {
            cout << recall << "\t" << time_us_per_query << " us\n";
            break;
        }
    }
}


template<typename DTres, typename DTset>
static void
test_vs_recall_multi_index(vector<HierarchicalNSW<DTres, DTset>*>& appr_alg_set,
                DTset *massQ, vector<vector<unsigned>>& massQA,
                map<string, size_t> &MapParameter,
                map<size_t, vector<float>>& mapResult) {
    size_t qsize = MapParameter["qsize"];
    size_t vecdim = MapParameter["vecdim"];
    size_t k = MapParameter["k"];
    int num_subgraph = MapParameter["num_subg"];

    vector<size_t> efs;
    // for (int i = 30; i < 100; i += 5)
    //     efs.push_back(i);
    efs.push_back(85);

    int column_map = 3;

    cout << "efs\t" << "R@" << k << "\t" << "time_us\t";
#if (RANKMAP && STAT)
        cout << "rank_us\t" << "sort_us\t" << "hlc_us\t" << "visited_us\t";
        cout << "NDC_max\t" << "NDC_total\t" << "n_hops\t";
        column_map = 9;
#else
    cout << "n_hop_L\t" << "n_hop_0\t" << "NDC\t";
#endif
    cout << endl;

    vector<float> result_ef(column_map);

    for (size_t ef : efs) {
        for (HierarchicalNSW<DTres, DTset>* appr_alg: appr_alg_set)
            appr_alg->setEf(ef);

#if (RANKMAP && STAT)
        appr_alg.stats->Reset();
#else
        for (HierarchicalNSW<DTres, DTset>* appr_alg: appr_alg_set) {
            appr_alg->metric_hops = 0;
            appr_alg->metric_distance_computations = 0;
        }
#endif

        vector<vector<unsigned>> result(qsize);
        for (vector<unsigned>& r: result)
            r.resize(k, 0);

        Timer stopw = Timer();
#if THREAD
#pragma omp parallel for
#endif
        for (int qi = 0; qi < qsize; qi++) {
#if RANKMAP
            priority_queue<pair<DTres, labeltype>> res = appr_alg.searchParaRank(massQ + vecdim * qi, k);
#else
            priority_queue<pair<DTres, labeltype>> res;
            for (HierarchicalNSW<DTres, DTset>* appr_alg: appr_alg_set) {
                priority_queue<pair<DTres, labeltype>> res_sub = appr_alg->searchKnn(massQ + vecdim * qi, k);
                if (res.empty())
                    res.swap(res_sub);
                else {
                    while (!res_sub.empty()) {
                        res.emplace(res_sub.top());
                        res.pop();

                        res_sub.pop();
                    }
                }
#if DDEBUG
                if (res.size() != k) {
                    printf("Error, res.size must is k\n"); exit(1);
                }
#endif
            }
#endif

#if THREAD
#pragma omp critical
#endif
            {
                int i = 0;
                while (!res.empty()){
                    result[qi][i] = (unsigned) res.top().second;
                    res.pop();
                    i++;
                }
            }
        }
        float time_us_per_query = stopw.getElapsedTimeus() / qsize / num_subgraph;
        float recall = comput_recall(result, massQA, qsize, k);

#if (RANKMAP && STAT)
        time_us_per_query = appr_alg.stats->hw_us / qsize;
#endif

        cout << ef << "\t" << recall << "\t" << time_us_per_query << "\t";
        size_t metric_hops = 0;
        size_t metric_distance_computations_total = 0;
        size_t metric_distance_computations_max = 0;
        for (HierarchicalNSW<DTres, DTset>* appr_alg: appr_alg_set) {
            metric_hops += appr_alg->metric_hops;
            metric_distance_computations_total += appr_alg->metric_distance_computations;
            metric_distance_computations_max = max<size_t>(metric_distance_computations_max, appr_alg->metric_distance_computations);
        }
        result_ef[0] = recall;
        result_ef[1] = time_us_per_query;
        result_ef[2] = 1.0 * metric_distance_computations_total / qsize;
#if (RANKMAP && STAT)
            cout << appr_alg.stats->all_rank_us / qsize << "\t";
            cout << appr_alg.stats->all_sort_us / qsize << "\t";
            cout << appr_alg.stats->all_hlc_us / qsize << "\t";
            cout << appr_alg.stats->all_visited_us / qsize << "\t";

            cout << (1.0 * appr_alg.stats->all_n_DC_max / qsize) << "\t";
            cout << (1.0 * appr_alg.stats->all_n_DC_total / qsize) << "\t";
            cout << (1.0 * appr_alg.stats->all_n_hops / qsize) << "\t";
#else
        cout << (1.0 * metric_hops / qsize) << "\t";
        cout << (1.0 * metric_distance_computations_total / qsize) << "\t";
        cout << (1.0 * metric_distance_computations_max / qsize) << "\t";
#endif
        cout << endl;

        mapResult.insert(pair<size_t, vector<float>>(ef, result_ef));

        if (recall > 1.0) {
            cout << recall << "\t" << time_us_per_query << " us\n";
            break;
        }
    }
}

inline bool exists_test(const string &name) {
    ifstream f(name.c_str());
    return f.good();
}

template<typename DTres, typename DTset>
void build_index(map<string, size_t> &MapParameter, map<string, string> &MapString,
                SpaceInterface<DTres, DTset> *l2space, bool isSave = true){
    //
    size_t efConstruction = MapParameter["efConstruction"];
    size_t M = MapParameter["M"];
    size_t vecsize = MapParameter["vecsize"];
    size_t vecdim = MapParameter["vecdim"];
    size_t qsize = MapParameter["qsize"];

    string path_data = MapString["path_data"];
    string index = MapString["index"];

    if (exists_test(index)){
        printf("Index %s is existed \n", index.c_str());
        return;
    } else {

        printf("Load base vectors: \n");
        ifstream base_reader(path_data.c_str());
        size_t head_offest = 2 * sizeof(uint32_t);
        size_t vec_offest = vecdim * sizeof(DTset);

        uint32_t nums_r, dims_r;
        base_reader.read((char *) &nums_r, sizeof(uint32_t));
        base_reader.read((char *) &dims_r, sizeof(uint32_t));
        uint32_t nums_ex = vecsize;
#if FROMBILLION
        nums_ex = 1e9;
#endif
        if ((nums_ex != nums_r) || (vecdim != dims_r)){
            printf("Error, file %s is error, nums_r: %u, dims_r: %u\n", path_data.c_str(), nums_r, dims_r);
            exit(1);
        }
        printf("vecsize: %lu, vecdim: %lu, path: %s\n", vecsize, vecdim, path_data.c_str());

        size_t build_start_id = 0;
        DTset* build_start_vector = new DTset[vecdim]();
#if PLATG
        DTset *massB = new DTset[vecsize * vecdim]();
        for (int i = 0; i < vecsize; i++)
            base_reader.read((char *) (massB + vecdim * i), vec_offest);
        build_start_id = compArrayCenter<DTset>(massB, vecsize, vecdim);
        delete[] massB;
#endif

        HierarchicalNSW<DTres, DTset> *appr_alg = new HierarchicalNSW<DTres, DTset>(l2space, vecsize, M, efConstruction);

        printf("Building index:\n");
        base_reader.seekg(head_offest + vec_offest * build_start_id, ios::beg);
        base_reader.read((char *) build_start_vector, vec_offest);
        appr_alg->addPoint((void *) build_start_vector, (size_t) build_start_id);

        int j1 = 1;
        StopW stopw = StopW();
        StopW stopw_full = StopW();
        size_t report_every = vecsize / 10;
#pragma omp parallel for
        for (size_t vi = 1; vi < vecsize; vi++) {
            unique_ptr<DTset> vecb(new DTset[vecdim]());
            size_t ic = vi;
#if PLATG
            if (vi <= build_start_id)
                ic = vi - 1;
#endif
#pragma omp critical
            {
                base_reader.seekg(head_offest + vec_offest * ic, ios::beg);
                base_reader.read((char *) vecb.get(), vec_offest);
                j1++;
                if (j1 % report_every == 0) {
                    cout << j1 * 10 / report_every << " %, "
                         << report_every / (1000.0 * stopw.getElapsedTimes()) << " kips " << " Mem: "
                         << getCurrentRSS() / 1000000 << " Mb \n";
                    stopw.reset();
                }
            }
            appr_alg->addPoint((void *) vecb.get(), ic);
        }
        base_reader.close();
        printf("Build time: %.3f seconds\n", stopw_full.getElapsedTimes());

        if (isSave)
            appr_alg->saveIndex(index);
        appr_alg->~HierarchicalNSW();

        printf("Build index %s is succeed \n", index.c_str());
    }
}

template<typename DTres, typename DTset>
void build_multi_index(map<string, size_t> &MapParameter, map<string, string> &MapString,
                SpaceInterface<DTres, DTset> *l2space, bool isSave = true){
    //
    size_t efConstruction = MapParameter["efConstruction"];
    size_t M = MapParameter["M"];
    size_t vecsize = MapParameter["vecsize"];
    size_t vecdim = MapParameter["vecdim"];
    size_t qsize = MapParameter["qsize"];

    string path_data = MapString["path_data"];
    string index = MapString["index"];

    if (exists_test(index)){
        printf("Index %s is existed \n", index.c_str());
        return;
    } else {

        printf("Load base vectors: \n");
        ifstream base_reader(path_data.c_str());
        size_t head_offest = 2 * sizeof(uint32_t);
        size_t vec_offest = vecdim * sizeof(DTset);

        uint32_t nums_r, dims_r;
        base_reader.read((char *) &nums_r, sizeof(uint32_t));
        base_reader.read((char *) &dims_r, sizeof(uint32_t));
        uint32_t nums_ex = vecsize;
#if FROMBILLION
        nums_ex = 1e9;
#endif
        if ((nums_ex != nums_r) || (vecdim != dims_r)){
            printf("Error, file %s is error, nums_r: %u, dims_r: %u\n", path_data.c_str(), nums_r, dims_r);
            exit(1);
        }
        printf("vecsize: %lu, vecdim: %lu, path: %s\n", vecsize, vecdim, path_data.c_str());

        DTset *massB = new DTset[vecsize * vecdim]();
        for (int i = 0; i < vecsize; i++)
            base_reader.read((char *) (massB + vecdim * i), vec_offest);

        int num_subgraph = MapParameter["num_subg"];

        AllocSubGraph allocSubGraph(MapString["dataname"], (int)vecsize, num_subgraph);
        allocSubGraph.computSubgCenter<DTset>(massB, vecdim);
        delete[] massB;

        for (int subg_i = 0; subg_i < num_subgraph; subg_i++) {
            size_t vecsize_subg = allocSubGraph.getSubgSize(subg_i);
            string sub_index = MapString["index_dir"] + "/" + to_string(subg_i) + ".bin";
            printf("Building sub-index %d:\n", subg_i);

            HierarchicalNSW<DTres, DTset>* appr_alg = new HierarchicalNSW<DTres, DTset>(l2space, vecsize_subg, M, efConstruction);

            size_t build_start_id = allocSubGraph.getSubgCenter(subg_i);
            DTset* build_start_vector = new DTset[vecdim]();
            base_reader.seekg(head_offest + vec_offest * build_start_id, ios::beg);
            base_reader.read((char *) build_start_vector, vec_offest);

            appr_alg->addPoint((void *) build_start_vector, (size_t) build_start_id);

            StopW stopw_full = StopW();
#pragma omp parallel for
            for (size_t vi = 1; vi < vecsize_subg; vi++) {
                unique_ptr<DTset> vecb(new DTset[vecdim]());
                size_t ic = allocSubGraph.getOriginId(subg_i, vi);
#pragma omp critical
                {
                    base_reader.seekg(head_offest + vec_offest * ic, ios::beg);
                    base_reader.read((char *) vecb.get(), vec_offest);
                }
                appr_alg->addPoint((void *) vecb.get(), ic);
            }

            if (appr_alg->cur_element_count != vecsize_subg) {
                printf("Error, cur_element_count is not equal size\n"); exit(1);
            }

            printf("Build time: %.3f seconds\n", stopw_full.getElapsedTimes());

            if (isSave)
                appr_alg->saveIndex(sub_index);
            appr_alg->~HierarchicalNSW();

            printf("Build sub_index %s is succeed \n", sub_index.c_str());
        }

        base_reader.close();
    }
}

template<typename DTres, typename DTset>
void search_index(map<string, size_t> &MapParameter, map<string, string> &MapString,
                SpaceInterface<DTres, DTset> *l2space){
    //
    size_t k = MapParameter["k"];
    size_t vecsize = MapParameter["vecsize"];
    size_t qsize = MapParameter["qsize"];
    size_t vecdim = MapParameter["vecdim"];
    size_t gt_maxnum = MapParameter["gt_maxnum"];

    string path_q = MapString["path_q"];
    string index = MapString["index"];
    string path_gt = MapString["path_gt"];

    if (!exists_test(index)){
        printf("Error, index %s is unexisted \n", index.c_str());
        exit(1);
    } else {

        vector<vector<unsigned>> massQA;
        DTset *massQ = new DTset[qsize * vecdim];

        printf("Loading GT:\n");
        LoadBinToVector<unsigned>(path_gt, massQA, qsize, gt_maxnum);
        printf("Loading queries:\n");
        LoadBinToArray<DTset>(path_q, massQ, qsize, vecdim);

#if SUBG
        int num_subgraph = MapParameter["num_subg"];
        vector<HierarchicalNSW<DTres, DTset> *> appr_alg_set(num_subgraph, nullptr);
        for (int subg_i = 0; subg_i < num_subgraph; subg_i++) {
            string sub_index = MapString["index_dir"] + "/" + to_string(subg_i) + ".bin";
            printf("Loading sub index from %s ...\n", sub_index.c_str());
            appr_alg_set[subg_i] = new HierarchicalNSW<DTres, DTset>(l2space, sub_index, false);
        }
#else
        printf("Loading index from %s ...\n", index.c_str());
        HierarchicalNSW<DTres, DTset> *appr_alg = new HierarchicalNSW<DTres, DTset>(l2space, index, false);
#endif

// #if THREAD
//         omp_set_num_threads(MapParameter["threads"]);
// #endif

#if RANKMAP
        appr_alg->initRankMap();
#endif

        vector<int> threadSet;
        for (int i = 1; i <= 8; i++)
            threadSet.push_back(i);
        for (int num_thread: threadSet) {

            omp_set_num_threads(num_thread);

            int test_times = TTIMES;
            printf("[Test Mode] Run %d times with %d threads and comput recall: \n", test_times, num_thread);
            vector<map<size_t, vector<float>>> finishedResult(test_times);
            for (int i = 0; i < test_times; i++) {
#if SUBG
            test_vs_recall_multi_index(appr_alg_set, massQ, massQA, MapParameter, finishedResult[i]);
#else
            test_vs_recall(*appr_alg, massQ, massQA, MapParameter, finishedResult[i]);
#endif
            }

            string result_file = MapString["result"] + to_string(num_thread) + ".log";
            ofstream result_writer(result_file.c_str());
            result_writer << "efs\t" << "R@" << k << "\t" << "time_us\t" << "NDC_total\t";
#if RANKMAP
            if (appr_alg->stats != nullptr) {
                result_writer << "rank_us\t" << "sort_us\t" << "hlc_us\t" << "visited_us\t";
                result_writer << "NDC_max\t" << "NDC_total\t" << "old_vst\t";
            }
#endif
            result_writer << "\n";

            for (auto iter = finishedResult[0].begin(); iter != finishedResult[0].end(); iter++) {
                size_t ef = iter->first;
                vector<float> recall_list(test_times);
                for (int ri = 0; ri < test_times; ri++){
                    recall_list[ri] = finishedResult[ri][ef][1];
                }

                int pos = 0;
                if (test_times >= 3)
                    pos = selectNearAvgPos(recall_list);

                vector<float> res_selected(finishedResult[pos][ef]);
                result_writer << ef << "\t";
                for (float& r : res_selected)
                    result_writer << r << "\t";
                result_writer << "\n";
            }
            result_writer.close();
            printf("Write final result to %s is successd\n", result_file.c_str());
        }

#if RANKMAP
        appr_alg->deleteRankMap();
#endif
#if SUBG
        // for (HierarchicalNSW<DTres, DTset>* appr_alg: appr_alg_set)
        //     appr_alg->~HierarchicalNSW();
        printf("Search sub index %s is succeed \n", MapString["index_dir"].c_str());
#else
        // appr_alg->~HierarchicalNSW();
        printf("Search index %s is succeed \n", index.c_str());
#endif

    }
}

void hnsw_impl(string stage, string using_dataset, size_t data_size_millions, size_t n_threads, size_t num_subgraph){
    string path_project = "..";
#if RANKMAP
    string label = "rank-map/";
#elif PLATG
    string label = "plat/";
#else
    string label = "hnsw/";
#endif
#if SUBG
    label = "sub-graph/";
#endif

    string path_graphindex = path_project + "/graphindex/" + label;
    createDir(path_graphindex);

    string pre_index = path_graphindex + using_dataset;
    createDir(pre_index);

    // for 1m, 10m, 100m
    vector<size_t> efcSet = {20, 30, 40};
    size_t M = (log10(data_size_millions) + 2) * 10;
	size_t efConstruction = M * 10;
    size_t k = 10;
    size_t vecsize = data_size_millions * 1000000;

    map<string, size_t> MapParameter;
    MapParameter["data_size_millions"] = data_size_millions;
    MapParameter["efConstruction"] = efConstruction;
    MapParameter["M"] = M;
    MapParameter["k"] = k;
    MapParameter["vecsize"] = vecsize;
#if BFMETIS
    if (data_size_millions == 50)
        MapParameter["M"] = 35;
    if (data_size_millions == 500 || data_size_millions == 1000)
        MapParameter["M"] = 40;
    MapParameter["efConstruction"] = MapParameter["M"] * 10;
#endif

    map<string, string> MapString;

#if SUBG
    string sub_index_dir = pre_index + "/" + using_dataset + to_string(data_size_millions) +
                        "m_ef" + to_string(MapParameter["efConstruction"]) + "m" + to_string(MapParameter["M"]) +
                        "_subg" + to_string(num_subgraph);
    createDir(sub_index_dir);
    MapParameter["num_subg"] = num_subgraph;
    MapString["index_dir"] = sub_index_dir;
    MapString["index"] = sub_index_dir + "/0.bin";
#else
    string hnsw_index = pre_index + "/" + using_dataset + to_string(data_size_millions) +
                        "m_ef" + to_string(MapParameter["efConstruction"]) + "m" + to_string(MapParameter["M"]) + ".bin";
    MapString["index"] = hnsw_index;
#endif

    CheckDataset(using_dataset, MapParameter, MapString);
#if THREAD
    MapParameter["threads"] = n_threads;
#endif

    string save_dir;
    string file_name = using_dataset + to_string(data_size_millions) + "m_rc" + to_string(k);
    string suffix = "";

#if SUBG
    file_name += "_sg" + to_string(num_subgraph);
#endif

#if PLATG
    save_dir = "plat_billion";
#else
    save_dir = "hnsw";
#endif

#if THREAD
    save_dir += "_multithread_multinode";
    suffix = "_t";
#endif

    save_dir = path_project + "/output/result/" + save_dir;
    createDir(save_dir);

    MapString["result"] = save_dir + "/" + file_name + suffix;


    if (stage == "build" || stage == "both") {
#if SUBG
        if (MapString["format"] == "Float") {
            L2Space l2space(MapParameter["vecdim"]);
            build_multi_index<float, float>(MapParameter, MapString, &l2space);
        } else if (MapString["format"] == "Uint8") {
            L2SpaceI<int, uint8_t> l2space(MapParameter["vecdim"]);
            build_multi_index<int, uint8_t>(MapParameter, MapString, &l2space);
        } else if (MapString["format"] == "Int8") {
            L2SpaceI<int, int8_t> l2space(MapParameter["vecdim"]);
            build_multi_index<int, int8_t>(MapParameter, MapString, &l2space);
        } else {
            printf("Error, unsupport format: %s \n", MapString["format"].c_str()); exit(1);
        }
#else
        if (MapString["format"] == "Float") {
            L2Space l2space(MapParameter["vecdim"]);
            build_index<float, float>(MapParameter, MapString, &l2space);
        } else if (MapString["format"] == "Uint8") {
            L2SpaceI<int, uint8_t> l2space(MapParameter["vecdim"]);
            build_index<int, uint8_t>(MapParameter, MapString, &l2space);
        } else if (MapString["format"] == "Int8") {
            L2SpaceI<int, int8_t> l2space(MapParameter["vecdim"]);
            build_index<int, int8_t>(MapParameter, MapString, &l2space);
        } else {
            printf("Error, unsupport format: %s \n", MapString["format"].c_str()); exit(1);
        }
#endif
    }

    if (stage == "search" || stage == "both") {
        if (MapString["format"] == "Float") {
            L2Space l2space(MapParameter["vecdim"]);
            search_index<float, float>(MapParameter, MapString, &l2space);
        } else if (MapString["format"] == "Uint8") {
            L2SpaceI<int, uint8_t> l2space(MapParameter["vecdim"]);
            search_index<int, uint8_t>(MapParameter, MapString, &l2space);
        } else if (MapString["format"] == "Int8") {
            L2SpaceI<int, int8_t> l2space(MapParameter["vecdim"]);
            search_index<int, int8_t>(MapParameter, MapString, &l2space);
        } else {
            printf("Error, unsupport format: %s \n", MapString["format"].c_str()); exit(1);
        }
    }

    return;
}
