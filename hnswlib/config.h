// For multi experiment
// #define DIR(A) #A

#define PLATG       true
#define PROFILE     false
#define MEMTRACE    false

#define MODE base

// typedef unsigned char DTSET;
// typedef unsigned char DTVAL;
// typedef int DTRES;
typedef float DTSET;
typedef float DTVAL;
typedef float DTRES;


// rank-level的映射
#define RANKMAP     true

#if RANKMAP
#define NUM_RANKS   1
#define RAM_RANK_GB 1
#define OPT_VISITED true
#endif