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

template<typename DTres, typename DTset>
void search_index(map<string, size_t> &index_parameter, map<string, string> &index_string,
                    SpaceInterface<DTres, DTset> *l2space, bool isTrans){
    //
    size_t vecsize = index_parameter["vecsize"];
    size_t vecdim = index_parameter["vecdim"];
    size_t qsize = index_parameter["qsize"];
    size_t gt_maxnum = index_parameter["gt_maxnum"];
    size_t k = gt_maxnum;

    string path_data = index_string["path_data"];
    string path_q = index_string["path_q"];
    string path_gt = index_string["path_gt"] + "_test";

    cout << "path_data: " << path_data.c_str() << "\n"
         << "path_q: " << path_q.c_str() << "\n"
         << "path_gt: " << path_gt.c_str() << "\n"
         << "trans?: " << isTrans << "\n"
         << "vecsize: " << vecsize << "\n"
         << "vecdim: " << vecdim << "\n";

    // 比较先后生成的两个base vector是否一致
#if TESTBASE
    {
        DTset *massB = new DTset[vecsize * vecdim];
        LoadBinToArrayIghead<DTset>(path_data, massB, vecsize, vecdim);

        DTset *massB_test = new DTset[vecsize * vecdim];
        string path_data_test = path_data + "_rewrite";
        LoadBinToArray<DTset>(path_data_test, massB_test, vecsize, vecdim);

        for (int i = 0; i < vecsize; i++){
            for (int j = 0; j < vecdim; j++){
                if (massB[i*vecdim+j] != massB_test[i*vecdim+j]){
                    printf("%d, %d: %.3f, %.3f\n", i, j, 1.0 * massB[i*vecdim+j], 1.0 * massB_test[i*vecdim+j]);
                }
                // printf("%d, %d: %.3f, %.3f\n", i, j, massB[i*vecdim+j], massB_test[i*vecdim+j]);
            }
            // exit(1);
        }
        exit(1);
    }
#endif

    // 比较先后生成的两个groundtruth是否一致
#if TESTGT
    {
        unsigned *massQA = new unsigned[qsize * gt_maxnum];
        LoadBinToArray<unsigned>(index_string["path_gt"], massQA, qsize, gt_maxnum);

        unsigned *massQA_test = new unsigned[qsize * gt_maxnum];
        LoadBinToArray<unsigned>(path_gt, massQA_test, qsize, gt_maxnum);

        for (int i = 0; i < qsize; i++){
            for (int j = 0; j < 10; j++){
                if (massQA[i*gt_maxnum+j] != massQA_test[i*gt_maxnum+j]){
                    printf("%d, %d: %d, %d\n", i, j, massQA[i*gt_maxnum+j], massQA_test[i*gt_maxnum+j]);
                }
            }
        }
        exit(1);
    }
#endif

    DTset *massB = new DTset[vecsize * vecdim]();
    cout << "Loading base data:\n";
    if (isTrans){
        LoadBinToArrayIghead<DTset>(path_data, massB, vecsize, vecdim);
        string path_test = path_data + "_rewrite";
        WriteBinToArray<DTset>(path_test, massB, vecsize, vecdim);
        // exit(1);
    } else
        LoadBinToArray<DTset>(path_data, massB, vecsize, vecdim);

    cout << "Building index:\n";
    BruteforceSearch<DTres, DTset>* brute_alg = new BruteforceSearch<DTres, DTset>(l2space, vecsize);
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
    int qsize_aligned = qsize / 10;
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
            if (ti % qsize_aligned == 0){
                int tt = ti / qsize_aligned;
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

    if (index_string["format"] == "Float") {
        L2Space l2space(index_parameter["vecdim"]);
        search_index<float, float>(index_parameter, index_string, &l2space, (isTrans == "trans"));
    } else if (index_string["format"] == "Uint8") {
        L2SpaceI<int, uint8_t> l2space(index_parameter["vecdim"]);
        search_index<int, uint8_t>(index_parameter, index_string, &l2space, (isTrans == "trans"));
    } else if (index_string["format"] == "Int8") {
        L2SpaceI<int, int8_t> l2space(index_parameter["vecdim"]);
        search_index<int, int8_t>(index_parameter, index_string, &l2space, (isTrans == "trans"));
    } else {
        printf("Error, unsupport format: %s \n", index_string["format"].c_str()); exit(1);
    }

    return;
}
