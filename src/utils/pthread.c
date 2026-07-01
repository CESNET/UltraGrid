// SPDX-License-Identifier: BSD-3-Clause
// Copyright (c) 2026 CESNET, zájmové sdružení právických osob
/**
 * @file
 * macOS doesn't have pthread_condattr_setclock() therefore CLOCK_MONOTONIC
 * cannot be set for pthread_cond_timedwait(). As CLOCK_REALTIME is the default,
 * it is not optimal (eventually non-continuous), use workaround on macOS
 * using sleep with timeout in relative time - pthread_cond_timedwait()
 * (duration).
 */
#include "utils/pthread.h"

#include <assert.h>  // for assert
#include <errno.h>   // for ETIMEDOUT
#include <stdio.h>   // for perror
#include <stdlib.h>  // for abort
#include <time.h>    // for timespec, CLOCK_MONOTONIC, clock_gettime

#ifdef HAVE_CONFIG_H
#include "config.h"     // for DEBUG
#endif
#include "tv.h"         // for NS_IN_SEC
#include "utils/misc.h" // for ug_strerror

#define MOD_NAME "[utils/pthread] "

/**
 * as pthread_mutex_init(). If DEBUG set, adds PTHREAD_MUTEX_ERRORCHECK attr
 */
void
ug_pthread_mutex_init(pthread_mutex_t *mutex)
{
        pthread_mutexattr_t attr;
        pthread_mutexattr_init(&attr);
#ifdef DEBUG
        pthread_mutexattr_settype(&attr, PTHREAD_MUTEX_ERRORCHECK);
#endif
        int rc = pthread_mutex_init(mutex, &attr);
        assert(rc == 0);
        pthread_mutexattr_destroy(&attr);
}

/**
 * initialize condition variable to be used with ug_pthread_cond_timedwait()
 *
 * destroy normally with pthread_cond_destroy()
 */
void
ug_pthread_cond_init(pthread_cond_t *cv)
{
        int ret = 0;
        pthread_condattr_t attr;
        pthread_condattr_init(&attr);
#if !defined __APPLE__ && !defined _WIN32
        ret = pthread_condattr_setclock(&attr, CLOCK_MONOTONIC);
        if (ret != 0) {
                (void) fprintf(stderr, "pthread_condattr_setclock: %s\n",
                               ug_strerror(ret));
                abort();
        }
#endif
        ret = pthread_cond_init(cv, &attr);
        if (ret != 0) {
                perror(__func__);
                abort();
        }
        pthread_condattr_destroy(&attr);
}

/**
 * wait for condition variable with timeout given by *timeout_ns
 *
 * Condition variable must be init ug_pthread_cond_init() otherwise the behavior
 * is undefined.
 *
 * Properties are the same as pthread_cond_timedwait() - return values,
 * possibility of spurious wake-ups.
 *
 * @param         cv          same as in pthred_cond_[timed]wait
 * @param         lock        same as in pthred_cond_[timed]wait
 * @param[in,out] timeout_ns  relative timeout for the CV wait - will be updated
 *                            with actual value (mainly for the case of spurious
 *                            wake-ups). Should not be negative on input and
 *                            is at least 0 on output.
 */
int
ug_pthread_cond_timedwait(pthread_cond_t *cv, pthread_mutex_t *lock,
                          time_ns_t *timeout_ns)
{
        struct timespec tmout = { 0, 0 };
        struct timespec t0;
        clock_gettime(CLOCK_MONOTONIC, &t0);
#if !defined __APPLE__ && !defined _WIN32
        // timeout in absolute time if not Mac
        clock_gettime(CLOCK_MONOTONIC, &tmout);
#endif
        long long nsec = tmout.tv_nsec + *timeout_ns;
        tmout.tv_sec += nsec / NS_IN_SEC;
        tmout.tv_nsec = nsec % NS_IN_SEC;
#if defined __APPLE__ || defined _WIN32
        int ret = pthread_cond_timedwait_relative_np(cv, lock, &tmout);
#else
        int ret = pthread_cond_timedwait(cv, lock, &tmout);
#endif
        struct timespec t1;
        clock_gettime(CLOCK_MONOTONIC, &t1);
        // adjust timeout
        *timeout_ns -=
            ((t1.tv_sec - t0.tv_sec) * NS_IN_SEC) + (t1.tv_nsec - t0.tv_nsec);
        if (*timeout_ns < 0) {
                *timeout_ns = 0;
        }
        if (ret != 0 && ret != ETIMEDOUT) {
                perror(__func__);
        }
        return ret;
}

