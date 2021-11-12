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

inline bool exists_test(const std::string &name) {
    ifstream f(name.c_str());
    return f.good();
}

template<typename DTset, typename DTval, typename DTres>
void build_index(const string &dataname, string &index, SpaceInterface<DTres> &s, size_t &efConstruction, 
                    size_t &M, size_t &vecsize, size_t &vecdim, string &path_data, bool isSave = true){
    // 
    if (exists_test(index)){
        printf("Index %s is existed \n", index.c_str());
        return;
    } else {
        DTset *massB = new DTset[vecsize * vecdim]();
        if (massB == nullptr){
            printf("Error, failed to allo mmeory\n");
            exit(1);
        }

        cout << "Loading base data:\n";
        ifstream inputB(path_data.c_str(), ios::binary);
        for (size_t i = 0; i < vecsize; i++){
            int expect_in;
            if (dataname == "sift" || dataname == "gist" || dataname == "deep"){
                inputB.read((char *) &expect_in, 4);
                if (expect_in != vecdim) {
                    cout << "file error";
                    exit(1);
                }
            }
            inputB.read((char *) (massB + i * vecdim), vecdim * sizeof(DTset));
        }
        inputB.close();

        cout << "Building index:\n";
        BruteforceSearch<DTres>* brute_alg = new BruteforceSearch<DTres>(&s, vecsize);
#pragma omp parallel for
        for (size_t i = 0; i < vecsize; i++){
            brute_alg->addPoint((void *) (massB + i * vecdim), i);
        }

        if (isSave)
            brute_alg->saveIndex(index);

        printf("Build index %s is succeed \n", index.c_str());
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
        BruteforceSearch<DTres> *brute_alg = new BruteforceSearch<DTres>(&s, index);

        unsigned *massQA = new unsigned[qsize * gt_maxnum]();
        DTres * massQA_dist = new DTres[qsize * gt_maxnum]();
        DTset *massQ = new DTset[qsize * vecdim]();
        
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

        for (size_t i = 0; i < qsize; i++){
            std::priority_queue<std::pair<DTres, labeltype >> rs = brute_alg->searchKnn(massQ + i * vecdim, gt_maxnum);
            size_t kk = k;
            while (!rs.empty()){
                massQA_dist[i * gt_maxnum + kk - 1] = rs.top().first;
                massQA[i * gt_maxnum + kk - 1] = rs.top().second;
                rs.pop();
                kk--;
            }
            if (kk != 0){
                printf("Error, expect nums: %u \n", k);
                exit(1);
            }
        }

        cout << "Writing GT:\n";
        string path_gt_id = path_gt + "id.bin";
        string path_gt_dist = path_gt + "dist.bin";
        WriteBinToArray<unsigned>(path_gt_id, massQA, qsize, gt_maxnum);
        WriteBinToArray<DTres>(path_gt_dist, massQA_dist, qsize, gt_maxnum);
        printf("Write GT to %s done \n", path_gt.c_str());

        printf("Search index %s is succeed \n", index.c_str());

    } 
}

void gene_gt_impl(bool is_build, const string &using_dataset){
    string prefix = "/home/ljun/anns/hnsw_nics/graphindex/";
    string label = "brute/";
    // if ()
    // support dataset: sift, gist, deep, glove, crawl
    // string using_dataset = "deep";

    string pre_index = prefix + label + using_dataset;
    if (access(pre_index.c_str(), R_OK|W_OK)){
        if (mkdir(pre_index.c_str(), S_IRWXU) != 0) {
            printf("Error, dir %s create failed \n", pre_index.c_str());
            exit(1);
        }
    }

	size_t subset_size_milllions = 100;
	size_t efConstruction = 40;
	size_t M = 16;
    size_t k = 10;
	
    size_t vecsize = subset_size_milllions * 1000;
    size_t qsize, vecdim, gt_maxnum;
    string path_index, path_gt, path_q, path_data;
    
    std::map<string, size_t> index_parameter;
    std::map<string, string> index_string;

    string hnsw_index = pre_index + "/" + using_dataset + to_string(subset_size_milllions) + 
                        "k_brute.bin";
    CheckDataset(using_dataset, index_parameter, index_string, subset_size_milllions, vecsize, qsize, vecdim, gt_maxnum,
                    path_q, path_data, path_gt);

    L2Space l2space(vecdim);
    gt_maxnum = k;
    path_gt = pre_index + "/" + using_dataset + to_string(subset_size_milllions) + "k_gt_";

    if (is_build){
        build_index<DTSET, DTVAL, DTRES>(using_dataset, hnsw_index, l2space, efConstruction, M, vecsize, vecdim, path_data);
    } else{
        search_index<DTSET, DTVAL, DTRES>(using_dataset, hnsw_index, l2space, k, qsize, vecdim, gt_maxnum, path_q, path_gt);
    }
    return;
}