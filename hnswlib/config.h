#define PLATG       true
#define FROMBILLION true

// rank-level的映射
#define RANKMAP     true

#if RANKMAP
#define NUM_RANKS   8
#define OPT_SORT    false
#define OPT_VISITED true
// rank 分配方式
#define MODMAP      true

// 需要设计拟合函数！消除计时引入的开销
#define STAT        true
#endif


#define DDEBUG      false

#define VHIT        true