#ifdef HAVE_CONFIG_H
#include "config.h"
#include "config_unix.h"
#include "config_win32.h"
#endif // HAVE_CONFIG_H


#ifdef WIN32

#include <time.h>
#include <windows.h>
#include <winbase.h>

#include "debug.h"
#include "gettimeofday.h"

static LARGE_INTEGER start_LI;
LARGE_INTEGER freq = {.QuadPart = 0};
static time_t start_time;

static void gettimeofday_init();

static void gettimeofday_init() {
	time_t tmp;
	HANDLE process;
	DWORD prio;

	process = GetCurrentProcess();
	prio = GetPriorityClass(process);
	int ret = SetPriorityClass(process, REALTIME_PRIORITY_CLASS);
	if(!ret) {
		fprintf(stderr, "SetPriorityClass failed.\n");
	}

	tmp = time(NULL);

	while( (start_time = time(NULL)) == tmp)
		;
	
	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&start_LI);
	SetPriorityClass(process, prio);
}

 
#if defined(_MSC_VER) || defined(_MSC_EXTENSIONS)
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000Ui64
#else
  #define DELTA_EPOCH_IN_MICROSECS  11644473600000000ULL
#endif
 
// Definition of a gettimeofday function
int gettimeofday_replacement(struct timeval *tv, void *tz)
{
	UNUSED(tz);
	LARGE_INTEGER val;

	if(freq.QuadPart == 0) {
		gettimeofday_init();
	}

	QueryPerformanceFrequency(&freq);
	QueryPerformanceCounter(&val);
	
	long long int difference = val.QuadPart - start_LI.QuadPart;

	tv->tv_sec =  start_time + difference / freq.QuadPart;
	tv->tv_usec = (long long)((double) difference * 1000*1000 / freq.QuadPart) % (1000 * 1000);

	return 0;
}

#endif
