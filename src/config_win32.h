/*
 * config-win32.h
 *
 * Windows specific definitions and includes.
 *
 * Copyright (c) 1995-2001 University College London
 * Copyright (c) 2001-2002 University of Southern California
 * 
 * Redistribution and use in source and binary forms, with or without
 * modification, is permitted provided that the following conditions
 * are met:
 * 
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 * 
 * 3. All advertising materials mentioning features or use of this software
 *    must display the following acknowledgement:
 * 
 *      This product includes software developed by the University of Southern
 *      California Information Sciences Institute.
 * 
 * 4. Neither the name of the University nor of the Institute may be used
 *    to endorse or promote products derived from this software without
 *    specific prior written permission.
 * 
 * THIS SOFTWARE IS PROVIDED BY THE AUTHORS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESSED OR IMPLIED WARRANTIES, INCLUDING,
 * BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY
 * AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
 * EVENT SHALL THE AUTHORS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
 * INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
 * OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
 * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 * $Revision: 1.1 $
 * $Date: 2007/11/08 09:48:59 $
 */
#define _WIN32_WINNT 0x0600
#ifdef WIN32
#ifndef _CONFIG_WIN32_H
#define _CONFIG_WIN32_H

// define compatibility version
#define __MSVCRT_VERSION__ 0x7000

#include <assert.h>
#include <process.h>
#include <malloc.h>
#include <stdio.h>
#include <memory.h>
#include <errno.h>
#include <math.h>
#include <stdlib.h>   /* abs() */
#include <string.h>

// 0x0501 is Win XP, 0x0502 2003 Server, 0x0600 Win Vista and Win 7 is 0x0601
#ifndef WINVER
#define WINVER 0x0600
#endif /* WINVER */
#ifndef _WIN32_WINNT
#define _WIN32_WINNT 0x0600
#endif /* _WIN32_WINNT */

#define WIN32_LEAN_AND_MEAN

#include <crtdefs.h>

#include <winsock2.h>
#include <ws2tcpip.h>
#include <naptypes.h>
#include <ntddndis.h>
#include <Iphlpapi.h> // if_nametoindex

#include <mmreg.h>
#include <mmsystem.h>
#include <windows.h>
#include <io.h>
#include <process.h>
#include <fcntl.h>
#include <time.h>

#include <amstream.h>

EXTERN_C const CLSID CLSID_NullRenderer;
EXTERN_C const CLSID CLSID_SampleGrabber;

typedef int		ttl_t;
typedef unsigned	fd_t;
typedef unsigned char	byte;
typedef int             ttl_t;
typedef unsigned int    fd_t;


/*
 * the definitions below are valid for 32-bit architectures and will have to
 * be adjusted for 16- or 64-bit architectures
 */
typedef unsigned int	in_addr_t;

#ifndef TRUE
#define FALSE	0
#define	TRUE	1
#endif /* TRUE */

#define USERNAMELEN	8
#define WORDS_SMALLENDIAN 1

#define NEED_INET_ATON

#include <time.h>		/* For clock_t */
#include "compat/usleep.h"

#define srand48	lbl_srandom
#define lrand48 lbl_random

typedef char	*caddr_t;

typedef struct iovec {
	caddr_t	iov_base;
	ssize_t	iov_len;
} iovec_t;

struct msghdr {
	caddr_t		msg_name;
	int		msg_namelen;
	struct iovec	*msg_iov;
	int		msg_iovlen;
	caddr_t		msg_accrights;
	int		msg_accrightslen;
};

#define MAXHOSTNAMELEN	256

#define SYS_NMLN	32
struct utsname {
	char sysname[SYS_NMLN];
	char nodename[SYS_NMLN];
	char release[SYS_NMLN];
	char version[SYS_NMLN];
	char machine[SYS_NMLN];
};

struct timezone;

typedef DWORD uid_t;
typedef DWORD gid_t;
    
#if defined(__cplusplus)
extern "C" {
#endif

// MinGW-w64 defines some broken macro for strtok_r in pthread.h
// which can be accidently included before this resulting in compilation
// error
#undef strtok_r

#ifndef HAVE_STRTOK_R
static inline char * strtok_r(char *str, const char *delim, char **save);

/*
 * Public domain licensed code taken from:
 * http://en.wikibooks.org/wiki/C_Programming/Strings#The_strtok_function
 */
static inline char *strtok_r(char *s, const char *delimiters, char **lasts)
{
     char *sbegin, *send;
     sbegin = s ? s : *lasts;
     sbegin += strspn(sbegin, delimiters);
     if (*sbegin == '\0') {
         /* *lasts = ""; */
         *lasts = sbegin;
         return NULL;
     }
     send = sbegin + strcspn(sbegin, delimiters);
     if (*send != '\0')
         *send++ = '\0';
     *lasts = send;
     return sbegin;
}
#endif

int uname(struct utsname *);
int getopt(int, char * const *, const char *);
//int strncasecmp(const char *, const char*, int len);
int srandom(int);
int random(void);
double drand48();
int gettimeofday(struct timeval *p, struct timezone *z);
unsigned int gethostid(void);
uid_t getuid(void);
gid_t getgid(void);
int   getpid(void);
int nice(int);
int usleep(unsigned int);
time_t time(time_t *);

const char * w32_make_version_info(char * rat_verion);

#define strcasecmp  _stricmp
#define strncasecmp _strnicmp

void ShowMessage(int level, char *msg);

#define bcopy(from,to,len) memcpy(to,from,len)

#if defined(__cplusplus)
}
#endif

#ifndef ECONNREFUSED
#define ECONNREFUSED	WSAECONNREFUSED
#endif
#ifndef ENETUNREACH
#define ENETUNREACH	WSAENETUNREACH
#endif
#ifndef EHOSTUNREACH
#define EHOSTUNREACH	WSAEHOSTUNREACH
#endif
#ifndef EWOULDBLOCK
#define EWOULDBLOCK	WSAEWOULDBLOCK
#endif
#ifndef EAFNOSUPPORT
#define EAFNOSUPPORT	WSAEAFNOSUPPORT
#endif

#define M_PI		3.14159265358979323846

#include <malloc.h>

#ifndef HAVE_ALIGNED_ALLOC
#define aligned_malloc _aligned_malloc
#define aligned_free _aligned_free
#endif

#ifdef HAVE_STDINT_H
#include <stdint.h>
#endif

// MinGW has this
#include <pthread.h>
#include <sys/stat.h>

// MinGW-w64 defines some broken macro for strtok_r in pthread.h
#undef strtok_r

#include "compat/inet_ntop.h"
#include "compat/inet_pton.h"
#include "compat/gettimeofday.h"
#define gettimeofday gettimeofday_replacement

#include <direct.h>
#define platform_mkdir _mkdir

// sysconf(_SC_NPROCESSORS_ONLN) substitution
#ifndef _SC_NPROCESSORS_ONLN
#define _SC_NPROCESSORS_ONLN 123456
static inline long sysconf_replacement(int name) {
        assert(name == _SC_NPROCESSORS_ONLN);
        SYSTEM_INFO info;
        GetSystemInfo(&info);
        return info.dwNumberOfProcessors;
}
#define sysconf sysconf_replacement
#endif

#endif 
#endif
