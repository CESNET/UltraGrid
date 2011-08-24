/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK
//
//    Provides basic thread functions for different OS
//
*/
#ifndef _DVS_THREAD_H
#define _DVS_THREAD_H

#ifdef WIN32
# include <windows.h>
#else
# include <pthread.h>
# include <sched.h>

# define THREAD_PRIORITY_LOWEST		0
# define THREAD_PRIORITY_NORMAL		1
# define THREAD_PRIORITY_TIME_CRITICAL	2

# ifndef TRUE
#  define TRUE 1
# endif

# ifndef FALSE
#  define FALSE 0
# endif
#endif

#include "dvs_setup.h"

#ifdef WIN32
typedef HANDLE              dvs_thread;
typedef CRITICAL_SECTION    dvs_mutex;
typedef HANDLE              dvs_cond;
#else
# define dvs_thread	pthread_t
# define dvs_mutex	pthread_mutex_t
typedef struct {
  pthread_cond_t cond;
  int count;
} dvs_cond;
#endif

/*** basic thread functions */
int  dvs_thread_create(dvs_thread * handle, void * (*function)(void *), void * arg, dvs_cond * finish);
void dvs_thread_exit(int * result, dvs_cond * finish);
int  dvs_thread_exitcode(dvs_thread * handle, dvs_cond * finish);
void dvs_thread_priority(dvs_thread * handle, int priority);

/*** basic locking functions */
void dvs_mutex_init(dvs_mutex * mutex);
void dvs_mutex_free(dvs_mutex * mutex);
void dvs_mutex_enter(dvs_mutex * mutex);
void dvs_mutex_leave(dvs_mutex * mutex);

/*** basic event processing functions */
void dvs_cond_init(dvs_cond * cond);
void dvs_cond_free(dvs_cond * cond);
void dvs_cond_wait(dvs_cond * cond, dvs_mutex * mutex, int locked);
void dvs_cond_broadcast(dvs_cond * cond, dvs_mutex * mutex, int locked);

#endif  /* !_DVS_THREAD_H */
