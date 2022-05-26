#define FMTINT      false

#if FMTINT
typedef uint8_t     DTSET;
typedef int         DTRES;
#else
typedef float       DTSET;
typedef float       DTRES;
#endif