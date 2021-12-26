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

void gene_gt_impl(bool is_build, const std::string &using_dataset, size_t &M_size);

int main(int argc, char **argv) {
    
    bool is_build;
    size_t M_size;

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

        M_size = atol(argv[3]);
    }

    gene_gt_impl(is_build, std::string(argv[2]), M_size);

    return 0;
};