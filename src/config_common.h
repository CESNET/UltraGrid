#ifndef CONFIG_COMMON_H_A4B25A33_74EC_435F_95DD_9738A7A23EA9
#define CONFIG_COMMON_H_A4B25A33_74EC_435F_95DD_9738A7A23EA9

#ifndef __cplusplus
#ifdef HAVE_STDBOOL_H
# include <stdbool.h>
#else
# ifndef HAVE__BOOL
#  ifdef __cplusplus
typedef bool _Bool;
#  else
#   define _Bool signed char
#  endif
# endif
# define bool _Bool
# define false 0
# define true 1
# define __bool_true_false_are_defined 1
#endif
#endif // ! defined __cplusplus

#undef MIN
#define MIN(a,b) (((a)<(b))?(a):(b))
#undef MAX
#define MAX(a,b) (((a)>(b))?(a):(b))

// VrgInputFormat::RGBA conflicts with codec_t::RGBA
#define RGBA VR_RGBA
#ifdef HAVE_VRG_H
#include <vrgstream.h>
#else
#include "vrgstream-fallback.h"
#endif
#undef RGBA

#endif // defined CONFIG_COMMON_H_A4B25A33_74EC_435F_95DD_9738A7A23EA9
