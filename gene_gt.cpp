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
        // edit size
        // {
        //     DTset *massB = new DTset[vecsize * vecdim]();
        //     if (massB == nullptr){
        //         printf("Error, failed to allo mmeory\n");
        //         exit(1);
        //     }

        //     std::ifstream file_reader(path_data.c_str(), ios::binary);
        //     uint32_t nums_r, dims_r;
        //     file_reader.read((char *) &nums_r, sizeof(uint32_t));
        //     file_reader.read((char *) &dims_r, sizeof(uint32_t));
        //     file_reader.read((char *) massB, vecsize * vecdim * sizeof(DTset));
        //     file_reader.close();
        //     WriteBinToArray<DTset>(path_data, massB, vecsize, vecdim);
        //     exit(0);
        // }

        DTval *massB = new DTval[vecsize * vecdim]();
        if (massB == nullptr){
            printf("Error, failed to allo mmeory\n");
            exit(1);
        }

        cout << "Loading base data:\n";
        if (dataname == "sift"){
            DTset *massB_set = new DTset[vecsize * vecdim]();
            LoadBinToArray<DTset>(path_data, massB_set, vecsize, vecdim);
            TransIntToFloat<DTset>(massB, massB_set, vecsize, vecdim);
            delete[] massB_set;
        } else {
            LoadBinToArray<DTval>(path_data, massB, vecsize, vecdim);
        }

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

        unsigned *massQA = new unsigned[qsize * gt_maxnum];
        DTval *massQ = new DTval[qsize * vecdim];
        
        cout << "Loading queries:\n";
        if (dataname == "sift"){
            DTset *massQ_set = new DTset[qsize * vecdim]();
            LoadBinToArray<DTset>(path_q, massQ_set, qsize, vecdim);
            TransIntToFloat<DTset>(massQ, massQ_set, qsize, vecdim);
            delete[] massQ_set;
        } else {
            LoadBinToArray<DTval>(path_q, massQ, qsize, vecdim);
        }

        printf("Load queries from %s done \n", path_q.c_str());

#pragma omp parallel for
        for (size_t i = 0; i < qsize; i++){
            std::priority_queue<std::pair<DTres, labeltype >> rs = brute_alg->searchKnn(massQ + i * vecdim, k);
            size_t kk = k;
            while (!rs.empty()){
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
        WriteBinToArray<unsigned>(path_gt, massQA, qsize, k);
        printf("Write GT to %s done \n", path_gt.c_str());

        printf("Search index %s is succeed \n", index.c_str());

    } 
}

void gene_gt_impl(bool is_build, const string &using_dataset, size_t &M_size){
    string prefix = "/home/usr-xkIJigVq/DataSet/" + using_dataset + "/";
    string label = "brute_index/";

    string pre_index = prefix + label;
    string pre_output = prefix + using_dataset + to_string(M_size) + "m/";
    if (access(pre_index.c_str(), R_OK|W_OK)){
        if (mkdir(pre_index.c_str(), S_IRWXU) != 0) {
            printf("Error, dir %s create failed \n", pre_index.c_str());
            exit(1);
        }
    }

	size_t subset_size_milllions = M_size;
	size_t efConstruction = 40;
	size_t M = 16;
    size_t k = 100;
	
    size_t vecsize = subset_size_milllions * 1000000;
    size_t qsize, vecdim, gt_maxnum;
    string path_index, path_gt, path_q, path_data;
    
    CheckDataset(using_dataset, subset_size_milllions, vecsize, qsize, vecdim, gt_maxnum,
                    path_q, path_data, path_gt);

    L2Space l2space(vecdim);

    string hnsw_index = pre_index + using_dataset + to_string(M_size) + "m_brute.bin";

    if (is_build){
        build_index<DTSET, DTVAL, DTRES>(using_dataset, hnsw_index, l2space, efConstruction, M, vecsize, vecdim, path_data);
    } else{
        search_index<DTSET, DTVAL, DTRES>(using_dataset, hnsw_index, l2space, k, qsize, vecdim, gt_maxnum, path_q, path_gt);
    }
    return;
}