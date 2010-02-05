#include "config_unix.h"
#include "config_win32.h"
#include "gettimeofday.h"

#if defined(NEED_GETTIMEOFDAY) && defined(WIN32)

int gettimeofday(struct timeval *tp, void *tz)
{
	struct _timeb timebuffer;   

	_ftime( &timebuffer );
	tp->tv_sec  = timebuffer.time;
	tp->tv_usec = timebuffer.millitm * 1000;
	return 0;
}

#endif
