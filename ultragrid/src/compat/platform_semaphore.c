#include "config.h"
#include "debug.h"

#ifdef HAVE_MACOSX
#include <mach/semaphore.h>
#include <mach/task.h>
#include <mach/mach.h>
#else
#include <semaphore.h>
#endif /* HAVE_MACOSX */

#include "compat/platform_semaphore.h"

void platform_sem_init(void * semStructure, int pshared, int initialValue)
{
    #ifdef HAVE_MACOSX
    UNUSED(pshared);
    semaphore_create(mach_task_self(), (semaphore_t *)semStructure, SYNC_POLICY_FIFO, initialValue);
    #else
    sem_init((sem_t *)semStructure, pshared, initialValue);
    #endif /* HAVE_MACOSX */
}

void platform_sem_post(void * semStructure)
{
    #ifdef HAVE_MACOSX
    semaphore_signal(*((semaphore_t *)semStructure));
    #else
    sem_post((sem_t *)semStructure);
    #endif /* HAVE_MACOSX */
}

void platform_sem_wait(void * semStructure)
{
    #ifdef HAVE_MACOSX
    semaphore_wait(*((semaphore_t *)semStructure));
    #else
    sem_wait((sem_t *)semStructure);
    #endif /* HAVE_MACOSX */
}
