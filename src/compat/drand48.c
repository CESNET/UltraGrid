#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif

#include "drand48.h"

#ifdef NEED_DRAND48

double drand48(void)
{
        unsigned int x = (rand() << 16) | rand();
        return ((double)x / (double)0xffffffff);
}

#endif
