#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

using namespace std;

// cpu_set_t  mask;
// inline void assignToThisCore(int core_id){
//     CPU_ZERO(&mask);
//     CPU_SET(core_id, &mask);
//     sched_setaffinity(0, sizeof(mask), &mask);
// }

void hnsw_impl(int stage, string &using_dataset, string &format, size_t &M_size, 
                size_t &efc, size_t &neibor, size_t &k_res, size_t &efs_max);

int main(int argc, char **argv) {
    
    int stage;

    if (std::string(argv[1]) == "build")
        stage = 0;
    else if (std::string(argv[1]) == "search")
        stage = 1;
    else if (std::string(argv[1]) == "both")
        stage = 2;
    else {
        printf("Usage: ./main [stage: search] [format] [dataset] [M_size] [efc] [M] [k]\n");
        exit(1);
    }

    string using_dataset = string(argv[2]);
    string format = string(argv[3]);
    size_t M_size = atol(argv[4]);
    size_t efc = atol(argv[5]);
    size_t neibor = atol(argv[6]);
    size_t k_res = atol(argv[7]);
    size_t efs_max = atol(argv[8]);
    
    hnsw_impl(stage, using_dataset, format, M_size, efc, neibor, k_res, efs_max);

    return 0;
};