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
#ifdef WIN32
#ifndef _CONFIG_WIN32_H
#define _CONFIG_WIN32_H

#include <assert.h>
#include <process.h>
#include <malloc.h>
#include <stdio.h>
#include <memory.h>
#include <errno.h>
#include <math.h>
#include <stdlib.h>   /* abs() */
#include <string.h>
#ifndef MUSICA_IPV6
#include <winsock2.h>
#endif

#ifdef HAVE_IPv6
#ifdef MUSICA_IPV6
#include <winsock6.h>
#else
#ifdef WIN2K_IPV6
#include <ws2tcpip.h>
#include <tpipv6.h>
#else
#include <ws2ip6.h>
#include <ws2tcpip.h>
#endif
#endif
#endif

#include <mmreg.h>
#include <msacm.h>
#include <mmsystem.h>
#include <windows.h>
#include <io.h>
#include <process.h>
#include <fcntl.h>
#include <time.h>

typedef int		ttl_t;
typedef unsigned	fd_t;
typedef unsigned char	byte;
typedef int             ttl_t;
typedef unsigned int    fd_t;


/*
 * the definitions below are valid for 32-bit architectures and will have to
 * be adjusted for 16- or 64-bit architectures
 */
typedef u_char		uint8_t;
typedef u_short		uint16_t;
typedef u_long		uint32_t;
typedef char		int8_t;
typedef short		int16_t;
typedef int		int32_t;
typedef __int64		int64_t;
typedef unsigned int	in_addr_t;

#ifndef TRUE
#define FALSE	0
#define	TRUE	1
#endif /* TRUE */

#define USERNAMELEN	8
#define WORDS_SMALLENDIAN 1

#define NEED_INET_ATON

#include <time.h>		/* For clock_t */
#include "usleep.h"

#define srand48	lbl_srandom
#define lrand48 lbl_random

#define IN_CLASSD(i)	(((int)(i) & 0xf0000000) == 0xe0000000)
#define IN_MULTICAST(i)	IN_CLASSD(i)

typedef char	*caddr_t;
typedef int	ssize_t;

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

struct timezone {
	int tz_minuteswest;
	int tz_dsttime;
};

typedef DWORD pid_t;
typedef DWORD uid_t;
typedef DWORD gid_t;
    
#if defined(__cplusplus)
extern "C" {
#endif

int uname(struct utsname *);
int getopt(int, char * const *, const char *);
int strncasecmp(const char *, const char*, int len);
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

int  RegGetValue(HKEY *, char *, char*, char*, int);
void ShowMessage(int level, char *msg);

#define bcopy(from,to,len) memcpy(to,from,len)

#if defined(__cplusplus)
}
#endif

#define ECONNREFUSED	WSAECONNREFUSED
#define ENETUNREACH	WSAENETUNREACH
#define EHOSTUNREACH	WSAEHOSTUNREACH
#define EWOULDBLOCK	WSAEWOULDBLOCK

#define M_PI		3.14159265358979323846

#endif 
#endif
