#define PLATG       true
#define PROFILE     false


#define FMTINT  false

#if FMTINT
typedef uint8_t DTSET;
typedef int     DTRES;
#else
typedef float   DTSET;
typedef float   DTRES;
#endif


// rank-level的映射
#define RANKMAP     true

#if RANKMAP
#define NUM_RANKS   1
#define OPT_VISITED true
#endif

// rank 分配方式
#define MODMAP      true