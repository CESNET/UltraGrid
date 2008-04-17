#include "config.h"

#ifdef HAVE_MACOSX
#include <mach/semaphore.h>
#include <mach/task.h>

typedef semaphore_t sem_t;
#else
#include <semaphore.h>
#endif /* HAVE_MACOSX */

inline void platform_sem_init(void * semStructure, int pshared, int initialValue);
inline void platform_sem_post(void * semStructure);
inline void platform_sem_wait(void * semStructure);
