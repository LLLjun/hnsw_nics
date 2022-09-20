#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

using namespace std;

cpu_set_t  mask;
inline void assignToThisCore(int core_id)
{
    CPU_ZERO(&mask);
    CPU_SET(core_id, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
}


void hnsw_impl(string stage, string using_dataset, size_t data_size, size_t n_threads, size_t num_subgraph);

int main(int argc, char **argv) {

    if (argc != 6){
        printf("Usage: ./main [stage: build or search or both] [dataset] [datasize] [thread] [num_subgraph]\n");
        exit(1);
    } else {
        if (string(argv[1]) != "build" && string(argv[1]) != "search" && string(argv[1]) != "both") {
            printf("[stage: build or search or both]\n");
            exit(1);
        }
    }

    string stage = string(argv[1]);
    size_t n_threads = atoi(argv[4]);
    size_t num_subgraph = atoi(argv[5]);
    // if (stage == "search" && n_threads == 1)
    //     assignToThisCore(0);

    hnsw_impl(stage, string(argv[2]), atoi(argv[3]), n_threads, num_subgraph);

    return 0;
};