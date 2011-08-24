/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK
//
//    Provides basic thread functions for different OS
//
*/

#include "../../header/dvs_thread.h"

int dvs_thread_create(dvs_thread * handle, void * (*function)(void *), void * arg, dvs_cond * finish)
{
#ifdef WIN32
  DWORD pid;
  if(finish) {
    *finish = CreateEvent(NULL, FALSE, FALSE, NULL);
  } 

  *handle = CreateThread(NULL, 0, (LPTHREAD_START_ROUTINE) function, arg, 0, &pid);

  return !(*handle == NULL);
#else
  if (pthread_create(handle, NULL, function, arg)) {
    return FALSE;
  } else {
    return TRUE;
  }
#endif
}

void dvs_thread_exit(int * result, dvs_cond * finish)
{
#ifdef WIN32
  if(finish) {
    SetEvent(*finish);
  } 

  ExitThread(*result);
#else
  pthread_exit(result);
#endif
}

int dvs_thread_exitcode(dvs_thread * handle, dvs_cond * finish)
{
#ifdef WIN32
  DWORD result;
  
  WaitForSingleObject(*finish, INFINITE);

  do {
    Sleep(10);
    if (!GetExitCodeThread(*handle, &result)) {
      return -1;
    }
  } while (result == STILL_ACTIVE);

  CloseHandle(*finish);

  return (int)result;
#else
  void * presult;

  /* this waits until thread termination and returns the exit code */
  if (pthread_join(*handle, &presult) == 0) {
    return *(int *)presult;
  } else {
    return -1;
  }
#endif
}

void dvs_thread_priority(dvs_thread * handle, int priority)
{
#ifdef WIN32
  SetThreadPriority(*handle, priority);
#elif !defined(__sun)
  {
    struct sched_param sp;

    /* We only use Round-Robin scheduling algorithm. Priority can be
       changed in three steps lowest(0), normal(50) and highest(99) */
    switch (priority) {
      case THREAD_PRIORITY_TIME_CRITICAL:
        sp.sched_priority = sched_get_priority_max(SCHED_RR);
        break;
      case THREAD_PRIORITY_NORMAL:
        sp.sched_priority = sched_get_priority_max(SCHED_RR) / 2;
        break;
      case THREAD_PRIORITY_LOWEST:
      default:
        sp.sched_priority = 0;
        break;
    }
    pthread_setschedparam(*handle, SCHED_RR, &sp);
  }
#endif
}

void dvs_mutex_init(dvs_mutex * mutex)
{
#ifdef WIN32
  InitializeCriticalSection(mutex);
#else
  pthread_mutex_init(mutex, NULL);
#endif
}

void dvs_mutex_free(dvs_mutex * mutex)
{
#ifdef WIN32
  DeleteCriticalSection(mutex);
#else
  pthread_mutex_destroy(mutex);
#endif
}

void dvs_mutex_enter(dvs_mutex * mutex)
{
#ifdef WIN32
  EnterCriticalSection(mutex);
#else
  pthread_mutex_lock(mutex);
#endif
}

void dvs_mutex_leave(dvs_mutex * mutex)
{
#ifdef WIN32
  LeaveCriticalSection(mutex);
#else
  pthread_mutex_unlock(mutex);
#endif
}

void dvs_cond_init(dvs_cond * cond)
{
#ifdef WIN32
  *cond = CreateEvent(NULL, FALSE, FALSE, NULL);
#else
  pthread_cond_init(&cond->cond, NULL);
  cond->count = 0;
#endif
}

void dvs_cond_free(dvs_cond * cond)
{
#ifdef WIN32
  CloseHandle(*cond);
#else
  pthread_cond_destroy(&cond->cond);
#endif
}

void dvs_cond_wait(dvs_cond * cond, dvs_mutex * mutex, int locked)
{
#ifdef WIN32
  if (locked) {
    LeaveCriticalSection(mutex);
  }

  WaitForSingleObject(*cond, INFINITE);

  if (locked) {
    EnterCriticalSection(mutex);
  }
#else
  if (!locked) {
    pthread_mutex_lock(mutex);
  }

  if(cond->count < 1) {
    /* mutex needs to be locked */
    pthread_cond_wait(&cond->cond, mutex);
  }

  if(cond->count > 0) {
    cond->count--;
  }

  if (!locked) {
    pthread_mutex_unlock(mutex);
  }
#endif
}

void dvs_cond_broadcast(dvs_cond * cond, dvs_mutex * mutex, int locked)
{
#ifdef WIN32
  if (locked) {
    LeaveCriticalSection(mutex);
  }

  SetEvent(*cond);

  if (locked) {
    EnterCriticalSection(mutex);
  }
#else
  if (!locked) {
    pthread_mutex_lock(mutex);
  }

  cond->count++;

  /* mutex needs to be locked */
  pthread_cond_broadcast(&cond->cond);

  if (!locked) {
    pthread_mutex_unlock(mutex);
  }
#endif
}
