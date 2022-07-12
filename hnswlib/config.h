#define PLATG       true
#define FROMBILLION true

// rank-level的映射
#define RANKMAP     true

#if RANKMAP
#define NUM_RANKS   1
#define OPT_SORT    false
#define OPT_VISITED false
// rank 分配方式
#define MODMAP      true
// 需要设计拟合函数！消除计时引入的开销
#define STAT        false

#define DDEBUG      false
#endif

#if (!RANKMAP)
// 测量原始的多线程版本时间
#define THREAD      false
#endif

// 支持测试模式
#define TESTMODE    true
#if TESTMODE
#define TTIMES      5
#define DETAIL      false
#endif
