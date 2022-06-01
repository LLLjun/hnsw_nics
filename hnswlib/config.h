#define PLATG       true
#define PROFILE     false
#define MEMTRACE    false
#define REPLACEQ    false


#define FMTINT  true

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
#define RAM_RANK_GB 1
#endif