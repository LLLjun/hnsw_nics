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

size_t efs_max = 300;
string path_root = "/home/ljun/anns/hnsw_nics/experiment/aware/for_train/";
string path_aware = path_root + "target_" + to_string(efs_max) + "/";


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
            vector<vector<float>> &hops){

    size_t correct = 0;
    size_t total = 0;
    //uncomment to test in parallel mode:
    //#pragma omp parallel for

    for (int i = 0; i < qsize; i++) {
        vector<float>().swap(hops[i]);
        // if (i == 449)
        //     continue;

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
            }
            result.pop();
        }

        for (float ten: appr_alg.res_tenth)
            hops[i].push_back(ten);
    }
    return 1.0f * correct / total;
}

/*
    input: defferent query's saerch result
    output: 0 is done, 1 need continue
*/
void getEveryQueryMinStep(size_t &qsize, vector<vector<float>> &dist_per_step_per_query, string &path_min_step,
                            size_t &step_target, string &path_recall_loss){
    unsigned *min_step = new unsigned[qsize]();
    unsigned *all_step = new unsigned[qsize]();
    unsigned *res_loss = new unsigned[qsize]();
    for (size_t i = 0; i < qsize; i++){
        all_step[i] = dist_per_step_per_query[i].size();
        float all_dist = dist_per_step_per_query[i].back();
        float cur_dist = dist_per_step_per_query[i].back();
        for (int j = all_step[i] - 1; j >= 0; j--){
            if (dist_per_step_per_query[i][j] > all_dist){
                min_step[i] = j + 1;
                break;
            }
        }

        for (int j = all_step[i] - 1; j >= step_target; j--){
            if (dist_per_step_per_query[i][j] > cur_dist){
                res_loss[i]++;
                cur_dist = dist_per_step_per_query[i][j];
            }
        }
    }
    string path_all_step = path_min_step + "_all_step.txt";
    WriteTxtToArray<unsigned>(path_all_step, all_step, qsize, 1);
    WriteTxtToArray<unsigned>(path_min_step, min_step, qsize, 1);
    WriteTxtToArray<unsigned>(path_recall_loss, res_loss, qsize, 1);
    printf("Generate per query min search step to %s done.\n", path_min_step.c_str());
    delete[] min_step;
    delete[] all_step;
}

template<typename DTval, typename DTres>
static void
test_vs_recall(DTval *massQ, size_t qsize, HierarchicalNSW<DTres> &appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<DTres, labeltype >>> &answers, size_t k) {
    vector<size_t> efs;// = { 10,10,10,10,10 };
    
    efs.push_back(efs_max);
    size_t step_target = (ITST + ITLE);
    vector<vector<float>> dist_per_step_per_query(qsize);
#if USESAMQ
    string path_min_step = path_aware + "train_min_step.txt";
    string path_recall_loss = path_aware + "train_recall_loss_" + to_string(step_target) + ".txt";
#else
    string path_min_step = path_aware + "test_min_step.txt";
    string path_recall_loss = path_aware + "test_recall_loss_" + to_string(step_target) + ".txt";
#endif
    cout << "ef\t" << "R@" << k << "\t" << "qps\t" << "hop_0\t" << "hop_L\n";
    for (size_t ef : efs) {

        appr_alg.setEf(ef);
        appr_alg.metric_hops = 0;
        appr_alg.metric_hops_L = 0;
        clk_get stopw = clk_get();

        float recall = test_approx(massQ, qsize, appr_alg, vecdim, answers, k, dist_per_step_per_query);

        float time_us_per_query = stopw.getElapsedTimeus() / qsize;
        float avg_hop_0 = 1.0f * appr_alg.metric_hops / qsize;
        float avg_hop_L = 1.0f * appr_alg.metric_hops_L / qsize;
        cout << ef << "\t" << recall << "\t" << 1e6 / time_us_per_query << "\t" 
        << avg_hop_0 << "\t" << avg_hop_L << "\n";

        // 生成每个query的最小步数，也就是label
        getEveryQueryMinStep(qsize, dist_per_step_per_query, path_min_step, step_target, path_recall_loss);

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
        if (access(path_aware.c_str(), R_OK|W_OK)){
            if (mkdir(path_aware.c_str(), S_IRWXU) != 0) {
                printf("Error, dir %s create failed \n", path_aware.c_str());
                exit(1);
            }
        }

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
        size_t iter_start = ITST;
        size_t iter_len = ITLE;
        size_t ovlp_len = OLLE;
        size_t num_stage = NMSG;
#if USESAMQ
        string path_train = path_aware + "train_" + to_string(iter_start) + "_" + to_string(iter_len) + 
                            "_" + to_string(ovlp_len) + "_" + to_string(num_stage) + ".txt";
#else
        string path_train = path_aware + "test_" + to_string(iter_start) + "_" + to_string(iter_len) + 
                            "_" + to_string(ovlp_len) + "_" + to_string(num_stage) + ".txt";
#endif
        ofstream train_data(path_train.c_str(), ios::trunc);
        vector<Lstm_Feature> SeqFeature;
        appr_alg->setEf(efs_max);

        train_data  << "q_id" << "\t" 
                    << "stage" << "\t"
                    << "cycle" << "\t"
                    << "dist_bound" << "\t"
                    << "dist_candi_top" << "\t"
                    << "dist_result_k" << "\t"
                    << "dist_result_1" << "\t"
                    << "diff_top" << "\t"
                    << "diff_top_k" << "\t"
                    << "diff_k_1" << "\t"
                    << "div_top_1" << "\t"
                    << "div_k_1" << "\t"
                    << "inter" << "\t"
                    << "remain_step" << endl;

        qsize = 100;
        for (size_t i = 0; i < qsize; i++){
            // if (i == 449)
            //     continue;
            appr_alg->createSequenceFeature(i, (massQ + i * vecdim), SeqFeature, k, iter_start, iter_len, ovlp_len, num_stage);
            for (Lstm_Feature fea_cur: SeqFeature){
                // if (i > 449)
                //     train_data << (fea_cur.q_id - 1) << "\t";
                // else
                train_data << fea_cur.q_id << "\t";
                train_data << fea_cur.stage << "\t";
                train_data << fea_cur.cycle << "\t";

                train_data << fea_cur.dist_bound << "\t";
                train_data << fea_cur.dist_candi_top << "\t";
                train_data << fea_cur.dist_result_k << "\t";
                train_data << fea_cur.dist_result_1 << "\t";

                train_data << fea_cur.diff_top << "\t";
                train_data << fea_cur.diff_top_k << "\t";
                train_data << fea_cur.diff_k_1 << "\t";

                train_data << fea_cur.div_top_1 << "\t";
                // train_data << fea_cur.div_k_1 << endl;
                train_data << fea_cur.div_k_1 << "\t";
                train_data << fea_cur.inter << "\t";

                train_data << fea_cur.remain_step << endl;
            }
            // train_data << SeqFeature[0].iscontinue << endl;
        }
        train_data.close();
        printf("Create train data to %s done \n", path_train.c_str());
#endif

#if GETMINSTEP
        cout << "Comput recall and generate min step: \n";
        test_vs_recall(massQ, qsize, *appr_alg, vecdim, answers, k);
#endif

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
    size_t qsize_kilo = 10;
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
