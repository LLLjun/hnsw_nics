#define PLATG       true
#define FROMBILLION false

// rank-level的映射
#define RANKMAP     false

#if RANKMAP
#define NUM_RANKS   1
#define OPT_SORT    false
#define OPT_VISITED false
// rank 分配方式
#define MODMAP      true
// 需要设计拟合函数！消除计时引入的开销
#define STAT        false
#endif

#if (!RANKMAP)
// 测量原始的多线程版本时间
#define THREAD      true
#endif

// 支持测试模式
#define TESTMODE    true
#if TESTMODE
#define TTIMES      7
#define DETAIL      false
#endif

#define PROEFS      true

#if PLATG
#define SUBG        false
#define SG_METIS    false
#endif

#define BFMETIS     true