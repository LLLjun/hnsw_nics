#define _GNU_SOURCE
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string>

cpu_set_t  mask;
inline void assignToThisCore(int core_id)
{
    CPU_ZERO(&mask);
    CPU_SET(core_id, &mask);
    sched_setaffinity(0, sizeof(mask), &mask);
}

void hnsw_impl(bool is_build, const std::string &using_dataset, std::string &graph_type);
// void gene_gt_impl(bool is_build, const std::string &using_dataset);

int main(int argc, char **argv) {
    
    bool is_build;
    std::string graph_type = "base";

    if (argc != 4){
        printf("Usage: ./main [stage: build or search] [dataset]\n");
        exit(1);
    } else {
        if (std::string(argv[1]) == "build")
            is_build = true;
        else if (std::string(argv[1]) == "search")
            is_build = false;
        else {
            printf("[stage: build or search]\n");
            exit(1);
        }
        
        graph_type = std::string(argv[3]);
        if ((graph_type != "knng") && (graph_type != "rng") && (graph_type != "base")) {
            printf("[graph type: knng, rng or base]\n");
            exit(1);
        }
    }
    // if (!is_build)
    //     assignToThisCore(19);
    
    hnsw_impl(is_build, std::string(argv[2]), graph_type);
    // gene_gt_impl(is_build, std::string(argv[2]));

    return 0;
};