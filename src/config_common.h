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

/// unconditional alternative to assert that is not affected by NDEBUG macro
#define UG_ASSERT(cond) do { if (!(cond)) { fprintf(stderr, "%s:%d: %s: Assertion `" #cond "' failed.\n", __FILE__, __LINE__, __func__); abort(); } } while(0)

#endif // defined CONFIG_COMMON_H_A4B25A33_74EC_435F_95DD_9738A7A23EA9
