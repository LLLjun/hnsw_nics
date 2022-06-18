#define PLATG       true


#define FMTINT  false

#if FMTINT
typedef uint8_t DTSET;
typedef int     DTRES;
#else
typedef float   DTSET;
typedef float   DTRES;
#endif


// rank-level的映射
#define RANKMAP     false

#if RANKMAP
#define NUM_RANKS   1
#define OPT_VISITED true
// rank 分配方式
#define MODMAP      true
#endif


// 支持测试模式
#define TESTMODE    true
#if TESTMODE
#define TTIMES      5
#define DETAIL      true
#endif