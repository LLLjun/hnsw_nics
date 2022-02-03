#define _GNU_SOURCE
#include <sched.h>
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
using namespace std::chrono;

cpu_set_t  mask;
inline void assignToThisCore(int core_id){
    CPU_ZERO(&mask);
    CPU_SET(core_id, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
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

template<typename DTres>
float comput_recall(std::vector<std::vector<unsigned>> &res, 
                vector<std::priority_queue<std::pair<DTres, labeltype >>> answers, size_t &qsize, size_t &k){
  size_t correct = 0;
  size_t total = 0;
  
  for (size_t qi = 0; qi < qsize; qi++){
    std::unordered_set<unsigned> g;
    while (answers[qi].size()){
        g.insert(answers[qi].top().second);
        answers[qi].pop();
    }
    total += res[qi].size();

    for (unsigned res_i : res[qi]){
      if (g.find(res_i) != g.end()){
        correct++;
      }
    }
  }
  return (float)correct / total;
}

template<typename DTval, typename DTres>
static void
test_vs_recall(DTval *massQ, size_t qsize, HierarchicalNSW<DTres> &appr_alg, size_t vecdim,
               vector<std::priority_queue<std::pair<DTres, labeltype >>> &answers, size_t k, string &log_file) {
    vector<size_t> efs;// = { 10,10,10,10,10 };

#if (MANUAL || MANUALRUN)
    efs.push_back(EFS_MAX);
    appr_alg.k_search = k;
#else
    if (k == 1){
        for (int i = 5; i <= 30; i += 5) {
            efs.push_back(i);
        }
        for (int i = 40; i <= 100; i += 10) {
            efs.push_back(i);
        }
        for (int i = 200; i <= 500; i += 100) {
            efs.push_back(i);
        }
    } else if (k == 10){
        for (int i = 20; i <= 200; i += 10) {
            efs.push_back(i);
        }
        for (int i = 200; i <= 1000; i += 100) {
            efs.push_back(i);
        }
    } else if (k == 100){
        for (int i = 110; i <= 400; i+= 20)
            efs.push_back(i);
        for (int i = 400; i <= 600; i += 50)
            efs.push_back(i);
        for (int i = 600; i <= 2000; i += 100)
            efs.push_back(i);
    }
#endif

    ofstream csv_writer(log_file.c_str(), ios::trunc);
#if NEWMETRIC
    csv_writer << "efs,R@" << k << ",nds" << endl;
#else
    csv_writer << "efs,R@" << k << ",qps" << endl;
#endif

    cout << "ef\t" << "R@" << k << "\t" << "qps\t" << "hop_0\t" << "hop_L\n";
    for (size_t ef : efs) {
        appr_alg.setEf(ef);
        appr_alg.metric_hops = 0;
        appr_alg.metric_hops_L = 0;
        appr_alg.metric_distance_computations = 0;

#if MANUALRUN
        map<string, float> param_aware;

        param_aware["len_observe"] = 50;
        param_aware["curve_x"] = 3;
        param_aware["dist_thr"] = 0.002;
        param_aware["std_thr"] = 0.01;
        param_aware["order_n"] = 3;
        param_aware["add_step"] = 6;

        appr_alg.setAwareParam(param_aware);
#endif

        std::vector<std::vector<unsigned>> res(qsize);

        steady_clock::time_point s = steady_clock::now();
        for (int i = 0; i < qsize; i++) {
            std::priority_queue<std::pair<DTres, labeltype >> result = appr_alg.searchKnn(massQ + vecdim * i, k);
            while (result.size()){
                res[i].push_back(result.top().second);
                result.pop();
            }
        }
        double time_ns_per_query = duration_cast<nanoseconds>(steady_clock::now() - s).count() / qsize;

        float recall = comput_recall(res, answers, qsize, k);

        float avg_hop_0 = 1.0f * appr_alg.metric_hops / qsize;
        float avg_hop_L = 1.0f * appr_alg.metric_hops_L / qsize;
        float avg_mds   = 1.0f * appr_alg.metric_distance_computations / qsize;

        cout << ef << "\t" << recall << "\t" << (1e9 / time_ns_per_query) << "\t" 
        << avg_hop_0 << "\t" << avg_hop_L << "\n";

#if NEWMETRIC
        csv_writer << ef << "," << recall << "," << avg_mds << endl;
#else
        csv_writer << ef << "," << recall << "," << (1e9 / time_ns_per_query) << endl;
#endif
        // if (recall > 0.98) {
        //     break;
        // }
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
    string format = index_string["format"];
    string path_build_txt = index_string["path_build_txt"];

    if (exists_test(index)){
        printf("Index %s is existed \n", index.c_str());
        return;
    } else {

        DTval *massB  = new DTval[vecsize * vecdim]();

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

        HierarchicalNSW<DTres> *appr_alg = new HierarchicalNSW<DTres>(&s, vecsize, M, efConstruction);
        // appr_alg->testSortMultiadd();
        // exit(0);
// #if MSH
//         // random
//         std::vector<uint32_t> rand_list(vecsize - 1);
//         for (uint32_t i = 1; i < vecsize; i++)
//             rand_list[i] = i;
//         random_shuffle(rand_list.begin(), rand_list.end());
// #endif

#if PLATG
        unsigned center_id = compArrayCenter<DTval>(massB, vecsize, vecdim);
        appr_alg->addPoint((void *) (massB + center_id * vecdim), (size_t) center_id);
#else
        appr_alg->addPoint((void *) (massB), (size_t) 0);
#endif
        cout << "Building index:\n";
        int j1 = 0;
        clk_get stopw = clk_get();
        size_t report_every = vecsize / 10;

        steady_clock::time_point s = steady_clock::now();
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

// #if MSH
//             appr_alg->addPoint((void *) (massB + rand_list[i] * vecdim), rand_list[i]);
// #else
            appr_alg->addPoint((void *) (massB + i * vecdim), i);
// #endif

#endif
        }

        double time_build = duration<double>(steady_clock::now() - s).count();

        cout << "Build time:" << time_build << "  seconds\n";
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
        float odg_avg = appr_alg->getDegreeDistri(dstb_in, dstb_out);

#if IOF1
        // IF1 -> IDG, OF1
        vector<unsigned> OF1, IF1, IDG, ODG;
        OF1.swap(appr_alg->OF1);
        IF1.swap(appr_alg->IF1);
        appr_alg->getDegreePerPoint(IDG, ODG);

        size_t maxM0 = 2 * M;

        vector<unsigned> OF1_dt(maxM0 + 1, 0);
        vector<unsigned> IF1_dt(maxM0 + 1, 0);
        for (size_t i = 0; i < vecsize; i++){
            if (OF1[i] <= maxM0)
                OF1_dt[OF1[i]]++;
            if (IF1[i] <= maxM0)
                IF1_dt[IF1[i]]++;
        }
        string path_OF1_dt = path_build_txt + "OF1_IF1_dt.csv";
        ofstream OF1_dt_writer(path_OF1_dt.c_str(), ios::trunc);
        for (size_t i = 0; i <= maxM0; i++){
            OF1_dt_writer << OF1_dt[i] << "," << IF1_dt[i] << endl;
        }
        OF1_dt_writer.close();

        vector<vector<unsigned>> IF1_to_IDG(maxM0 + 1);
        for (size_t i = 0; i <= maxM0; i++)
            IF1_to_IDG[i].resize(maxM0 + 1, 0);
        for (size_t i = 0; i < vecsize; i++){
            // unsigned row = IF1[i] > maxM0 ? maxM0: IF1[i];
            // unsigned col = IDG[i] > maxM0 ? maxM0: IDG[i];
            // IF1_to_IDG[row][col]++;
            if (IF1[i] <= maxM0 && IDG[i] <= maxM0)
                IF1_to_IDG[IF1[i]][IDG[i]]++;
        }
        string path_IF1_to_IDG = path_build_txt + "IF1_to_IDG.csv";
        ofstream IF1_to_IDG_writer(path_IF1_to_IDG.c_str(), ios::trunc);
        for (size_t i = 0; i <= maxM0; i++){
            for (size_t j = 0; j <= maxM0; j++)
                IF1_to_IDG_writer << ((float) IF1_to_IDG[i][j] / IF1_dt[i]) << ",";
            IF1_to_IDG_writer << endl;
        }
        IF1_to_IDG_writer.close();
#endif

        // write to build txt
        ofstream txt_writer(path_build_txt.c_str(), ios::trunc);
        txt_writer << "Build time: " << time_build << " seconds\n";
#if PROFILE
        float part_total = (appr_alg->tb_search + appr_alg->tb_sort);
        txt_writer << "Search time: " << appr_alg->tb_search << " seconds, Percent: " << (appr_alg->tb_search / part_total) << endl;
        txt_writer << "Sort time: " << appr_alg->tb_sort << " seconds, Percent: " << (appr_alg->tb_sort / part_total) << endl;
        float gap_multi_thread = part_total / 112 / time_build;
        txt_writer << "Similarity: " << gap_multi_thread << endl;
#endif
        txt_writer << "Average degree: " << odg_avg << endl;
        txt_writer.close();
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
    string format = index_string["format"];

    string path_search_csv = index_string["path_search_csv"];

    if (!exists_test(index)){
        printf("Error, index %s is unexisted \n", index.c_str());
        exit(1);
    } else {

        HierarchicalNSW<DTres> *appr_alg = new HierarchicalNSW<DTres>(&s, index, false);

        unsigned *massQA = new unsigned[qsize * gt_maxnum];
        DTval *massQ = new DTval[qsize * vecdim];

        cout << "Loading GT:\n";
        LoadBinToArray<unsigned>(path_gt, massQA, qsize, gt_maxnum);
        printf("Load GT from %s done \n", path_gt.c_str());
        
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
        printf("Load queries from %s done \n", path_q.c_str());
    
#if MSH
        vector<size_t> efs_list;
        efs_list.push_back(100);
        efs_list.push_back(200);
        efs_list.push_back(300);
        for (size_t efs: efs_list){
            string path_step = index_string["prefix_unique"] + "_step_efs" + to_string(efs) + ".csv";
            appr_alg->setEf(efs);
            appr_alg->getEveryQueryMinStep(massQ, qsize, vecdim, k, path_step);
        }

        // {        
        //     size_t qid = 4397;
        //     string path_trace = index_string["prefix_unique"] + "_trace_" + to_string(qid) + ".csv";
        //     appr_alg->getSearchTrace((massQ + qid * vecdim), k, path_trace);
        // }
        exit(0);
#endif

        vector<std::priority_queue<std::pair<DTres, labeltype >>> answers;
        cout << "Parsing gt:\n";
        get_gt(massQA, qsize, gt_maxnum, vecdim, answers, k);

#if MANUAL
        appr_alg->setEf(index_parameter["efs_max"]);
        appr_alg->k_search = k;
        bool debug = false;

        string path_train_result = index_string["prefix_param"] + 
                                "_efs" + to_string(index_parameter["efs_max"]) + ".csv";
        map<string, float> param;
        vector<map<string, float>> param_list;

        if (debug){
            param["len_observe"] = 18;
            param["curve_x"] = 4;
            param["dist_thr"] = 1.3;
            param["std_thr"] = 0.04;
            param["order_n"] = 3;
            param["add_step"] = 5;

            param_list.push_back(param);
        } else {
            for (size_t len = 20; len <= 50; len += 5){
                for (float cx = 1; cx <= 3; cx += 0.5){
                    for (float d_thr = 0.002; d_thr <= 0.02; d_thr += 0.002){
                        for (float s_thr = 0.005; s_thr <= 0.05; s_thr += 0.005){
                            for (size_t od = 3; od <= 5; od += 1){
                                for (size_t as = 0; as <= 6; as += 3){
                                    param["len_observe"] = len;
                                    param["curve_x"] = cx;
                                    param["dist_thr"] = d_thr;
                                    param["std_thr"] = s_thr;
                                    param["order_n"] = od;
                                    param["add_step"] = as;

                                    param_list.push_back(param);
                                }
                            }
                        }
                    }
                }
            }
        }

        appr_alg->predictStopByManual(massQ, qsize, vecdim, k, param_list, path_train_result, debug);


        // param["len_observe"] = 18;
        // param["dist_thr"] = 1.3;
        // param["std_thr"] = 0.04;
        // param["order_n"] = 3;
        // param["add_step"] = 5;


        // appr_alg->predictStopByManual(massQ, qsize, vecdim, k, param);
#endif
#if (MANUALRUN || (!MANUAL && !MANUALRUN))
        cout << "Comput recall: \n";
        test_vs_recall(massQ, qsize, *appr_alg, vecdim, answers, k, path_search_csv);
#endif

#if CREATESF
        size_t iter_start = ITST;
        size_t iter_len = ITLE;
        size_t ovlp_len = OLLE;
        size_t num_stage = NMSG;
#if USESAMQ
        string path_train = index_string["prefix_unique"] + "_train_" + to_string(iter_start) + "_" + to_string(iter_len) + 
                            "_" + to_string(ovlp_len) + "_" + to_string(num_stage) + ".csv";
#else
        string path_train = index_string["prefix_unique"] + "_test_" + to_string(iter_start) + "_" + to_string(iter_len) + 
                            "_" + to_string(ovlp_len) + "_" + to_string(num_stage) + ".csv";
#endif
        ofstream train_data(path_train.c_str(), ios::trunc);
        vector<Lstm_Feature> SeqFeature;
        appr_alg->setEf(EFS_MAX);

        train_data  << "q_id" << "," 
                    << "stage" << ","
                    << "cycle" << ","

                    << "dist_candi_top" << ","
                    << "dist_result_k" << ","
                    << "dist_result_1" << ","

                    << "diff_top" << ","
                    << "diff_top_k" << ","
                    << "diff_k_1" << ",";

                    for (size_t ii = 0; ii < appr_alg->maxM0_; ii++)
                        train_data << "ng_" + to_string(ii) << ",";

                    train_data << "remain_step" << endl;

        // qsize = 1000;
        for (size_t i = 0; i < qsize; i++){
            // if (i == 449)
            //     continue;
            appr_alg->createSequenceFeature(i, (massQ + i * vecdim), SeqFeature, k, iter_start, iter_len, ovlp_len, num_stage);
            for (Lstm_Feature fea_cur: SeqFeature){

                train_data << fea_cur.q_id << ",";
                train_data << fea_cur.stage << ",";
                train_data << fea_cur.cycle << ",";

                train_data << fea_cur.dist_candi_top << ",";
                train_data << fea_cur.dist_result_k << ",";
                train_data << fea_cur.dist_result_1 << ",";

                train_data << fea_cur.diff_top << ",";
                train_data << fea_cur.diff_top_k << ",";
                train_data << fea_cur.diff_k_1 << ",";

                for (size_t ii = 0; ii < appr_alg->maxM0_; ii++)
                    train_data << fea_cur.dist_candi_top_neig[ii] << ",";

                train_data << fea_cur.remain_step << endl;
            }
            // train_data << SeqFeature[0].iscontinue << endl;
        }
        train_data.close();
        printf("Create train data to %s done \n", path_train.c_str());
#endif



        printf("Search index %s is succeed \n", index.c_str());
        delete[] massQA;
        delete[] massQ;
    }
}

void hnsw_impl(int stage, string &using_dataset, string &format, size_t &M_size, 
                size_t &efc, size_t &neibor, size_t &k_res, size_t &efs_max){
    string root_index = "/home/usr-xkIJigVq/vldb/hnsw_nics/graphindex/";
    string root_output = "/home/usr-xkIJigVq/vldb/hnsw_nics/output/";

    string label = "expc3/";
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
    string pre_param = root_output + label + using_dataset + "/param";
    if (access(pre_param.c_str(), R_OK|W_OK)){
        if (mkdir(pre_param.c_str(), S_IRWXU) != 0) {
            printf("Error, dir %s create failed \n", pre_param.c_str());
            exit(1);
        }
    }

	size_t subset_size_milllions = M_size;
	size_t efConstruction = efc;
	size_t M = neibor;
    size_t k = k_res;

    string unique_name_b = using_dataset + to_string(subset_size_milllions) + 
                        "m_ef" + to_string(efConstruction) + "_M" + to_string(M);
#if EXI
    unique_name_b += "_sxi";
#endif

    string unique_name_s = unique_name_b + "_k" + to_string(k);
#if USESAMQ
    unique_name_s += "_samq";
#endif

    size_t vecsize = subset_size_milllions * 1000000;
    size_t qsize, vecdim, gt_maxnum;
    string path_index, path_gt, path_q, path_data;
    
    string hnsw_index = pre_index + "/" + unique_name_b + ".bin";
    string path_build_txt = pre_output + "/" + unique_name_b + "_build.txt";
    string path_search_csv = pre_output + "/" + unique_name_s + "_search.csv";

    map<string, size_t> index_parameter;
    index_parameter["subset_size_milllions"] = subset_size_milllions;
    index_parameter["efConstruction"] = efConstruction;
    index_parameter["M"] = M;
    index_parameter["k"] = k;
    index_parameter["vecsize"] = vecsize;
    index_parameter["efs_max"] = efs_max;

    map<string, string> index_string;
    index_string["using_dataset"] = using_dataset;
    index_string["format"] = format;
    index_string["hnsw_index"] = hnsw_index;
    index_string["path_build_txt"] = path_build_txt;
    index_string["path_search_csv"] = path_search_csv;
    index_string["prefix_param"] = pre_param + "/" + unique_name_s;
    index_string["prefix_unique"] = pre_output + "/" + unique_name_s;

    CheckDataset(using_dataset, index_parameter, index_string);
    
    L2Space l2space(index_parameter["vecdim"]);


    if (stage == 0 || stage == 2){
        if (format == "float")
            build_index<float, DTVAL, DTRES>(using_dataset, l2space, index_parameter, index_string);
        else if (format == "uint8")
            build_index<uint8_t, DTVAL, DTRES>(using_dataset, l2space, index_parameter, index_string);
    }
    
    if (stage == 1 || stage == 2){
#if (!NEWMETRIC)
        assignToThisCore(27);
#endif
        if (format == "float")
            search_index<float, DTVAL, DTRES>(using_dataset, l2space, index_parameter, index_string);
        else if (format == "uint8")
            search_index<uint8_t, DTVAL, DTRES>(using_dataset, l2space, index_parameter, index_string);
    }
    return;
}
