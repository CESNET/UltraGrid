/*
 * config-win32.h
 *
 * Windows specific definitions and includes.
 *
 * Copyright (c) 1995-2001 University College London
 * Copyright (c) 2001-2002 University of Southern California
 * Copyright (c) 2004-2023 CESNET, z. s. p. o.
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
#ifdef __MINGW32__
#ifndef _CONFIG_WIN32_H
#define _CONFIG_WIN32_H

//// define compatibility version - rather do not define it (if really needed, check if it works for both MSCVRT and UCRT)
//#ifndef __MSVCRT_VERSION__
//#define __MSVCRT_VERSION__ 0x700
//#endif

// 0x0501 is Win XP, 0x0502 2003 Server, 0x0600 Win Vista and Win 7 is 0x0601
#ifndef WINVER
#define WINVER _WIN32_WINNT_WIN7
#endif /* WINVER */
#ifndef _WIN32_WINNT
#define _WIN32_WINNT _WIN32_WINNT_WIN7
#endif /* _WIN32_WINNT */

#define WIN32_LEAN_AND_MEAN

#include <assert.h>
#include <process.h>
#include <malloc.h>
#include <stdio.h>
#include <memory.h>
#include <errno.h>
#include <math.h>
#include <stdlib.h>   /* abs() */
#include <string.h>

#include <crtdefs.h>

#include <winsock2.h>
#include <ws2tcpip.h>
#include <naptypes.h>
#include <ntddndis.h>
#include <iphlpapi.h> // if_nametoindex

#include <mmreg.h>
#include <mmsystem.h>
#include <windows.h>
#include <io.h>
#include <process.h>
#include <fcntl.h>
#include <time.h>

#include <amstream.h>

typedef int		ttl_t;
typedef unsigned char	byte;
typedef SOCKET          fd_t;


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
#include "compat/alarm.h"
#include "compat/usleep.h"

typedef char	*caddr_t;

#ifndef  _MSC_VER
typedef struct iovec {
	caddr_t	iov_base;
	ssize_t	iov_len;
} iovec_t;
#endif

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

int uname(struct utsname *);
int getopt(int, char * const *, const char *);
//int strncasecmp(const char *, const char*, int len);
unsigned int gethostid(void);
uid_t getuid(void);
gid_t getgid(void);
int   getpid(void);
int nice(int);

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

#define aligned_malloc _aligned_malloc
#define aligned_free _aligned_free

#ifdef HAVE_STDINT_H
#include <stdint.h>
#endif

// MinGW has this
#ifndef _MSC_VER
#include <pthread.h>
#include <sys/stat.h>
#endif

#define SHUT_RD SD_RECEIVE
#define SHUT_WR SD_SEND
#define SHUT_RDWR SD_BOTH

#define inet_pton InetPtonA
#define inet_ntop InetNtopA

#if _M_IX86_FP == 2
#undef __SSE2__
#define __SSE2__
#endif

#define sleep(sec) Sleep(1000 * (sec))

#define CLOSESOCKET closesocket

#endif // defined _CONFIG_WIN32_H

#endif // defined __MINGW32__
