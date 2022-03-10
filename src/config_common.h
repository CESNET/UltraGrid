#ifndef CONFIG_COMMON_H_A4B25A33_74EC_435F_95DD_9738A7A23EA9
#define CONFIG_COMMON_H_A4B25A33_74EC_435F_95DD_9738A7A23EA9

#ifndef __cplusplus
# include <stdbool.h>
#endif // ! defined __cplusplus

#undef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#undef MAX
#define MAX(a,b) (((a)>(b))?(a):(b))

#ifndef EXTERN_C
#ifdef __cplusplus
#define EXTERN_C extern "C"
#else
#define EXTERN_C
#endif
#endif // defined EXTERN_C

#endif // defined CONFIG_COMMON_H_A4B25A33_74EC_435F_95DD_9738A7A23EA9
