#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H


#ifdef WIN32

#include <time.h>
#include "debug.h"
#include "gettimeofday.h"

// Definition of a gettimeofday function
int gettimeofday_replacement(struct timeval *tv, void *tz)
{
	UNUSED(tz);
        struct timespec ts;
        timespec_get(&ts, TIME_UTC);

        tv->tv_sec = ts.tv_sec;
        tv->tv_usec = ts.tv_nsec / 1000;

	return 0;
}

#endif
