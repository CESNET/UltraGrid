// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 CESNET, zájmové sdružení právických osob

#ifndef UTILS_PTHREAD_COND_T_E0632D04_6CB6_41F5_AC7F_E49076756E78
#define UTILS_PTHREAD_COND_T_E0632D04_6CB6_41F5_AC7F_E49076756E78

#include <pthread.h>

#include "tv.h"  // for time_ns_t

void ug_pthread_cond_init(pthread_cond_t *cv);
int  ug_pthread_cond_timedwait(pthread_cond_t *cv, pthread_mutex_t *lock,
                               time_ns_t *timeout_ns);

#endif // defined UTILS_PTHREAD_COND_T_E0632D04_6CB6_41F5_AC7F_E49076756E78
