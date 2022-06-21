#define PLATG       false


#define FMTINT  true

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
#define NUM_RANKS   4
#define OPT_VISITED false
// rank 分配方式
#define MODMAP      true
#endif


// 支持测试模式
#define TESTMODE    true
#if TESTMODE
#define TTIMES      10
#define DETAIL      false
#endif