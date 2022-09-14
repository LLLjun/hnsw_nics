#define PLATG       true
#define FROMBILLION true

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


#define DDEBUG      false

#define PROEFS      true

#define PARTGRAPH   true
#if PARTGRAPH
#define UNWEIGHT    true
#endif

#define QTRACE      false