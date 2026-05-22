// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 CESNET, zájmové sdružení právických osob

#ifndef UTILS_PTHREAD_COND_T_E0632D04_6CB6_41F5_AC7F_E49076756E78
#define UTILS_PTHREAD_COND_T_E0632D04_6CB6_41F5_AC7F_E49076756E78

#include <pthread.h>

#include "debug.h"      // for MSG
#include "tv.h"         // for time_ns_t
#include "utils/misc.h" // IWYU pragma: keep for ug_streror

// PTHREAD_NULL compat
#ifndef PTHREAD_NULL // defined by POSIX v8
        #define PTHREAD_NULL ((pthread_t) { 0 })
#endif

#define CHK_PTHR(cmd)                                                          \
        do {                                                                   \
                int rc = cmd;                                                  \
                if (rc != 0) {                                                 \
                        MSG(ERROR, "%s:%d: " #cmd ": %s\n", __func__,          \
                            __LINE__, ug_strerror(rc));                        \
                }                                                              \
        } while (0)

void ug_pthread_mutex_init(pthread_mutex_t *mutex);
void ug_pthread_cond_init(pthread_cond_t *cv);
int  ug_pthread_cond_timedwait(pthread_cond_t *cv, pthread_mutex_t *lock,
                               time_ns_t *timeout_ns);

#endif // defined UTILS_PTHREAD_COND_T_E0632D04_6CB6_41F5_AC7F_E49076756E78
