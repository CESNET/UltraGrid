/*
//    Part of the DVS (http://www.dvs.de) *DVS*VERSION* SDK 
//
//    This is the sv/svram program. Here you can find the source
//    code that is using most of the setup functions or control 
//    functions of DVS DDR or DVS Videocards.
//
*/

#ifndef _SVPROGRAM_H
#define _SVPROGRAM_H

#include "../../header/dvs_setup.h"

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <sys/types.h>

#include <fcntl.h>
#include <string.h>
#include <errno.h>
#if !defined(linux) && !defined(WIN32) && !defined (__CYGWIN__)
#  include <sys/lock.h>
#endif
#include <signal.h>
#include <math.h>
#if defined(WIN32)
#  include <io.h>
#  include <winsock.h>
#  include <windows.h>
#  include <process.h>
#else
#  include <unistd.h>
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <arpa/inet.h>
#endif

#include "../../header/dvs_fm.h"
#include "../../header/dvs_clib.h"
#include "../../header/dvs_fifo.h"

#include "svtags.h"
#include "../common/dvs_support.h"

#define PRONTO_VERSION_MAJOR    2
#define PRONTO_VERSION_MINOR    3
#define PRONTO_VERSION_PATCH    22

#ifdef WIN32
/* the following part is copied from the mv hostsoftware  301199 WS */

#define MV_SHM_TEMPLATE         "DVS_SV_Map_%d"

typedef struct {
  int     magic;                  /* feed beef    */
  int     major;                  /* major rev.   */
  int     minor;                  /* minor rev.   */
  int     size;                   /* length       */
  int     sigcode;                /* signal code  */
  char    reserved1[1024-20];     /* header etc.  */
  /* more to come */
} MVSHM;

HANDLE  hMvShm;
MVSHM   *pMvShm;
#endif	/* WIN32 */

#endif
