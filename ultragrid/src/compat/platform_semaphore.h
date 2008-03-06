#include "config.h"

#ifdef HAVE_MACOSX
#include <mach/semaphore.h>
#include <mach/task.h>

typedef semaphore_t sem_t;
#endif /* HAVE_MACOSX */

void platform_sem_init(void * semStructure, int pshared, int initialValue);
void platform_sem_post(void * semStructure);
void platform_sem_wait(void * semStructure);
