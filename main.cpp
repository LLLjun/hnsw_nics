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

void hnsw_impl(const std::string &using_dataset, size_t sizeVectorM, std::string isTrans);

int main(int argc, char **argv) {

    if (argc != 4){
        printf("Usage: ./main [dataset] [size(Million)] [trans?]\n");
        exit(1);
    }

    hnsw_impl(std::string(argv[1]), std::atoi(argv[2]), std::string(argv[3]));

    return 0;
};