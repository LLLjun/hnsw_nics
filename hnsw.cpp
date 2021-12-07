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

#if (USESAMQ || CREATESF)
string path_aware = "/home/ljun/anns/hnsw_nics/experiment/aware/other_sample/";
#else
// string path_aware = "/home/ljun/anns/hnsw_nics/experiment/aware/exi/efc40m16/";
string path_aware = "/home/ljun/anns/hnsw_nics/experiment/aware/other_sample/test_gt/";
#endif

template<typename DTres>
void handleIndegree(HierarchicalNSW<DTres> &appr_alg, unsigned qi, const void *query, vector<labeltype> &gt_list, string &flag){
    string path_aware = "/home/ljun/anns/hnsw_nics/experiment/aware/exi/efc40m16/";
    string path_stroe = path_aware + "query" + to_string(qi) + "_" + flag + ".log";

    float dist_sq;
    std::vector<std::vector<Node_Connect_Info>> in_connect;
    size_t find_step = 1;
    appr_alg.getParentDistri(query, gt_list, find_step, in_connect, dist_sq);

    std::ofstream file_writer(path_stroe.c_str(), ios::trunc);
    file_writer << "dist start to query: " << dist_sq << endl;
    file_writer << endl;
    for (size_t st_i = 0; st_i < find_step; st_i++){
        file_writer << "find step: " << st_i <<endl;
        for (size_t ch_i = 0; ch_i < in_connect[st_i].size(); ch_i++){
            file_writer << "child: " << in_connect[st_i][ch_i].node_self << "\t";
            file_writer << "to query: " << in_connect[st_i][ch_i].dist_self_query << "\t";
            file_writer << "to start: " << in_connect[st_i][ch_i].dist_self_start << endl;
            for (size_t pa_i = 0; pa_i < in_connect[st_i][ch_i].node_parent.size(); pa_i++){
                file_writer << in_connect[st_i][ch_i].node_parent[pa_i] << "\t"
                            << in_connect[st_i][ch_i].dist_parent_query[pa_i] << "\t"
                            << in_connect[st_i][ch_i].dist_parent_start[pa_i] << endl;
            }
            file_writer << endl;
        }
        file_writer << endl;
    }
    file_writer.close();
    printf("Write data to %s done.\n", path_stroe.c_str());
}

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
            vector<std::priority_queue<std::pair<DTres, labeltype >>> &answers, size_t k, 
#if GETPERHOP
            unsigned *hops){
#else
            float *recall_cur = nullptr, float *gt_mean = nullptr, float *gt_sdev = nullptr) {
#endif
    size_t correct = 0;
    size_t total = 0;
    vector<float> dist_c;
    //uncomment to test in parallel mode:
    //#pragma omp parallel for
#if ANAYQ
    {
        int i = 160;
        vector<labeltype> gt_true;
        vector<labeltype> gt_false;

#else
    for (int i = 0; i < qsize; i++) {
#endif
#if GETMINEFS
        if (i == 449)
            continue;
#endif
        size_t cor_i = 0;
#if GETPERHOP
        appr_alg.metric_hops = 0;
#endif
        std::priority_queue<std::pair<DTres, labeltype >> result = appr_alg.searchKnn(massQ + vecdim * i, k);
        std::priority_queue<std::pair<DTres, labeltype >> gt(answers[i]);
        unordered_set<labeltype> g;
        total += gt.size();

        while (gt.size()) {
            g.insert(gt.top().second);
            gt.pop();
        }

#if GETMINEFS
            vector<float>().swap(dist_c);
#endif

        while (result.size()) {
            if (g.find(result.top().second) != g.end()) {
                correct++;
                cor_i++;
#if ANAYQ
                gt_true.push_back(result.top().second);
            } else {}
                // gt_false.push_back(result.top().second);
#else
            }
#endif
#if GETMINEFS
            dist_c.push_back(result.top().first);
#endif
            result.pop();
        }
#if GETMINEFS
        if (recall_cur != nullptr)
            recall_cur[i] = (float)cor_i / k;

        if (gt_mean != nullptr){
            float tt = 0;
            for (float dd: dist_c)
                tt += dd;
            gt_mean[i] = tt / dist_c.size();
            float ss = 0;
            for (float dd: dist_c)
                ss += (dd - gt_mean[i]) * (dd - gt_mean[i]);
            gt_sdev[i] = sqrtf32(ss);
        }
#endif

#if ANAYQ
        string flag;
        if (!gt_true.empty()){
            flag = "true";
            handleIndegree<DTres>(appr_alg, i, massQ + vecdim * i, gt_true, flag);
        }
        string path_search_dist = path_aware + "query" + to_string(i) + "_efs" + to_string(appr_alg.ef_) + "_dist.log";
        ofstream file_dist(path_search_dist.c_str());
        file_dist << "candicate top" << endl;
        for (float cat: appr_alg.candi_top)
            file_dist << cat << endl;
        file_dist << endl;
        file_dist << "bound" << endl;
        for (float cat: appr_alg.bound)
            file_dist << cat << endl;
        file_dist << endl;
        file_dist << "result first" << endl;
        for (float cat: appr_alg.res_first)
            file_dist << cat << endl;
        file_dist << endl;
        file_dist << "result tenth" << endl;
        for (float cat: appr_alg.res_tenth)
            file_dist << cat << endl;
        file_dist.close();

        exit(0);
#endif
#if GETPERHOP
        hops[i] = appr_alg.metric_hops;
#endif

    }
    return 1.0f * correct / total;
}

/*
    input: defferent efs's saerch result
    output: 0 is done, 1 need continue
*/
void getEveryQueryMinEfs(size_t &k, size_t &qsize, unsigned *mp_total, float *recall_target, unordered_set<size_t> &fixed, size_t &efs_last, float *recall_cur){
    for (size_t i = 0; i < qsize; i++){
        if (fixed.find(i) == fixed.end()){
            if (recall_cur[i] < recall_target[i]){
                fixed.insert(i);
                mp_total[i] = efs_last;
            }
        }
    }
}

template<typename DTval, typename DTres>
static void
test_vs_recall(DTval *massQ, size_t qsize, HierarchicalNSW<DTres> &appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<DTres, labeltype >>> &answers, size_t k) {
    vector<size_t> efs;// = { 10,10,10,10,10 };
    // for (int i = k; i < 30; i++) {
    //     efs.push_back(i);
    // }
    
#if GETMINEFS
    // for (int i = 1; i < 10; i++) {
    //     efs.push_back(i);
    // }
    // for (int i = 10; i < 100; i += 10) {
    //     efs.push_back(i);
    // }
    // for (int i = 100; i < 500; i += 40) {
    //     efs.push_back(i);
    // }
    // for (int i = 500; i < 1500; i += 200) {
    //     efs.push_back(i);
    // }
    for (int i = 10; i <= 300; i += 3) {
        efs.push_back(i);
    }
    
    unsigned *mp_total = new unsigned[qsize]();
    float *recall_target = new float[qsize]();
    float *dist_gt_mean = new float[qsize]();
    float *dist_gt_sdev = new float[qsize]();

    float *recall_cur = new float[qsize]();
    unordered_set<size_t> fixed;
    for (size_t i = 0; i < qsize; i++)
        mp_total[i] = efs[0];
    
    for (int i = efs.size() - 1; i >= 0; i--){
        size_t ef = efs[i];
#elif (ANAYQ || GETPERHOP)
    efs.push_back(1500);
    unsigned *hops = new unsigned[qsize]();
    for (size_t ef : efs) {
#else
    for (int i = 40; i < 100; i += 10) {
        efs.push_back(i);
    }
    for (int i = 100; i < 500; i += 40) {
        efs.push_back(i);
    }
    for (int i = 500; i < 1500; i += 200) {
        efs.push_back(i);
    }
    cout << "ef\t" << "R@" << k << "\t" << "qps\t" << "hop_0\t" << "hop_L\n";
    for (size_t ef : efs) {
#endif
        appr_alg.setEf(ef);
        appr_alg.metric_hops = 0;
        appr_alg.metric_hops_L = 0;
        clk_get stopw = clk_get();

#if GETMINEFS
        float recall = test_approx(massQ, qsize, appr_alg, vecdim, answers, k, recall_cur, dist_gt_mean, dist_gt_sdev);
#elif GETPERHOP
        float recall = test_approx(massQ, qsize, appr_alg, vecdim, answers, k, hops);
#else
        float recall = test_approx(massQ, qsize, appr_alg, vecdim, answers, k);
#endif
        float time_us_per_query = stopw.getElapsedTimeus() / qsize;
        float avg_hop_0 = 1.0f * appr_alg.metric_hops / qsize;
        float avg_hop_L = 1.0f * appr_alg.metric_hops_L / qsize;

#if GETMINEFS
        if (i == (efs.size() - 1)){
            memcpy(recall_target, recall_cur, qsize * sizeof(float));
            string path_gt_mean = path_aware + "recall@" + to_string(k) + "_gt_mean.log";
            string path_gt_sdev = path_aware + "recall@" + to_string(k) + "_gt_sdev.log";
            WriteTxtToArray<float>(path_gt_mean, dist_gt_mean, qsize, 1);
            WriteTxtToArray<float>(path_gt_sdev, dist_gt_sdev, qsize, 1);
            delete[] dist_gt_mean;
            delete[] dist_gt_sdev;
            dist_gt_mean = nullptr;
            dist_gt_sdev = nullptr;
        } else {
            getEveryQueryMinEfs(k, qsize, mp_total, recall_target, fixed, efs[i+1], recall_cur);
        }

        cout << ef << "\t" << recall << "\t" << 1e6 / time_us_per_query << "\t" 
        << avg_hop_0 << "\t" << avg_hop_L << "\t" << fixed.size() << "\n";

        if ((fixed.size() == qsize) || (i == 0)){
            printf("All queries get its min efs: %u.\n", ef);
            string path_recall_value = path_aware + "recall@" + to_string(k) + "_value.log";
            string path_min_efs = path_aware + "recall@" + to_string(k) + "_min_efs.log";
            WriteTxtToArray<float>(path_recall_value, recall_target, qsize, 1);
            WriteTxtToArray<unsigned>(path_min_efs, mp_total, qsize, 1);
            break;
        }
#elif GETPERHOP
        string path_hop = path_aware + "perhop_efs" + to_string(ef) + ".bin";
        WriteTxtToArray<unsigned>(path_hop, hops, qsize, 1);
#else
        cout << ef << "\t" << recall << "\t" << 1e6 / time_us_per_query << "\t" 
        << avg_hop_0 << "\t" << avg_hop_L << "\n";
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
void build_index(const string &dataname, string &index, SpaceInterface<DTres> &s, size_t &efConstruction, 
                    size_t &M, size_t &vecsize, size_t &vecdim, string &path_data, string &graph_type, bool isSave = true){
    // 
    if (exists_test(index)){
        printf("Index %s is existed \n", index.c_str());
        return;
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

        HierarchicalNSW<DTres> *appr_alg = new HierarchicalNSW<DTres>(&s, vecsize, M, efConstruction);
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
void search_index(const string &dataname, string &index, SpaceInterface<DTres> &s, size_t &k, 
                    size_t &qsize, size_t &vecdim, size_t &gt_maxnum, string &path_q, string &path_gt){
    // 
    if (!exists_test(index)){
        printf("Error, index %s is unexisted \n", index.c_str());
        exit(1);
    } else {

        unsigned *massQA = new unsigned[qsize * gt_maxnum];
        DTset *massQ = new DTset[qsize * vecdim];

        cout << "Loading GT:\n";
#if USESAMQ
        LoadBinToArray<unsigned>(path_gt, massQA, qsize, gt_maxnum);
#else
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
#endif
        printf("Load GT from %s done \n", path_gt.c_str());
        
        cout << "Loading queries:\n";
#if USESAMQ
        LoadBinToArray<DTset>(path_q, massQ, qsize, vecdim);
#else
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
#endif
        printf("Load queries from %s done \n", path_q.c_str());

        HierarchicalNSW<DTres> *appr_alg = new HierarchicalNSW<DTres>(&s, index, false);
    
        vector<std::priority_queue<std::pair<DTres, labeltype >>> answers;
        cout << "Parsing gt:\n";
        get_gt(massQA, qsize, gt_maxnum, vecdim, answers, k);

#if CREATESF
        size_t iter_start = 30;
        size_t iter_len = 50;
#if USESAMQ
        string path_train = path_aware + "train_" + to_string(iter_start) + "_" + to_string(iter_len) + ".txt";
#else
        string path_train = path_aware + "test_" + to_string(iter_start) + "_" + to_string(iter_len) + ".txt";
#endif
        ofstream train_data(path_train.c_str(), ios::trunc);
        vector<Lstm_Feature> SeqFeature;

        appr_alg->setEf(300);
        for (size_t i = 0; i < qsize; i++){
            if (i == 449)
                continue;
            appr_alg->createSequenceFeature(i, (massQ + i * vecdim), SeqFeature, k, iter_start, iter_len);
            for (Lstm_Feature fea_cur: SeqFeature){
                if (i > 449)
                    train_data << (fea_cur.q_id - 1) << "\t";
                else
                    train_data << fea_cur.q_id << "\t";
                train_data << fea_cur.cycle << "\t";
                train_data << fea_cur.dist_candi_top << "\t";
                train_data << fea_cur.dist_result_k << "\t";
                train_data << fea_cur.dist_result_1 << "\t";
                train_data << fea_cur.dist_div_top_k << "\t";
                train_data << fea_cur.dist_div_k_1 << endl;
                // train_data << fea_cur.isinter << endl;
            }
        }
        train_data.close();
        printf("Create train data to %s done \n", path_train.c_str());
#else
        cout << "Comput recall: \n";
        test_vs_recall(massQ, qsize, *appr_alg, vecdim, answers, k);
#endif
        // // get degree info
        // vector<size_t> dstb_in;
        // vector<size_t> dstb_out;
        // appr_alg->getDegreeDistri(dstb_in, dstb_out);

        printf("Search index %s is succeed \n", index.c_str());
    }
}

void hnsw_impl(bool is_build, const string &using_dataset, string &graph_type){
    string prefix = "/home/ljun/anns/hnsw_nics/graphindex/";

    string label = "profile/aware/exi/";

    // support dataset: sift, gist, deep, glove, crawl

    string pre_index = prefix + label + using_dataset;
    if (access(pre_index.c_str(), R_OK|W_OK)){
        if (mkdir(pre_index.c_str(), S_IRWXU) != 0) {
            printf("Error, dir %s create failed \n", pre_index.c_str());
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
    CheckDataset(using_dataset, subset_size_milllions, vecsize, qsize, vecdim, gt_maxnum,
                    path_q, path_data, path_gt);

    L2Space l2space(vecdim);

#if USESAMQ
    size_t qsize_kilo = 100;
    qsize = qsize_kilo * 1000;
    string path_lstm = "/home/ljun/anns/hnsw_nics/graphindex/lstm";
    string path_sample = path_lstm + "/" + using_dataset + "/" + 
                    using_dataset + to_string(subset_size_milllions) + "m_other_sample" + to_string(qsize_kilo) + "k_";
    path_q = path_sample + "query.bin";
    path_gt = path_sample + "gt_id.bin";
#endif

    if (is_build){
        build_index<DTSET, DTVAL, DTRES>(using_dataset, hnsw_index, l2space, efConstruction, M, vecsize, vecdim, path_data, graph_type);
    } else{
        search_index<DTSET, DTVAL, DTRES>(using_dataset, hnsw_index, l2space, k, qsize, vecdim, gt_maxnum, path_q, path_gt);
    }
    return;
}
