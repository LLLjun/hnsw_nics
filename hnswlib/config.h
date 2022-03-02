// For multi experiment
// #define DIR(A) #A

#define PLATG true

#define MODE base

// typedef unsigned char DTSET;
// typedef unsigned char DTVAL;
// typedef int DTRES;
typedef float DTSET;
typedef float DTVAL;
typedef float DTRES;
typedef int8_t DTFDRAM;
// typedef int16_t DTFSSD;

// 计算
typedef int8_t  FCP8;
typedef int16_t FCP16;
typedef int32_t FCP32;
typedef int64_t FCP64;


#define NUM_BANKS 8
#define NUM_PERSPNODE 1
#define NUM_CLUSTER_TRAIN 4e5

#define USEFIX false
#define USEHSG true

#if USEFIX
typedef int16_t DTFSSD;
#else
typedef float DTFSSD;
#endif

#if USEHSG
#define NUM_HSG_NODE 1e5
#define NUM_HSG_K 20
#endif

// 非均匀量化的倍数，目前需要手动给
#define PORP 2
#define EFS_PROP 0.5