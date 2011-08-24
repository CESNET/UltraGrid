/*
//    Part of the DVS SDK (http://www.dvs.de)
//
//    dpxrender - Shows the usage of the Render API.
//
*/


#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <fcntl.h>
#include <string.h>
#include <signal.h>

#ifdef WIN32
#  include <windows.h>
#  include <io.h>
#  pragma warning ( disable : 4100 )
#  pragma warning ( disable : 4244 )
#  pragma warning ( disable : 4127 )
#  pragma warning ( disable : 4702 )
#else
#  ifdef linux
#    define __USE_LARGEFILE64
#  endif
#  include <unistd.h>
#  define MAX_PATH	1024
#endif

#if !defined(WIN32)
#  define ERROR_IO_PENDING              EAGAIN
#  define QueryPerformanceCounter(x)
#endif

#include "../../header/dvs_setup.h"
#include "../../header/dvs_clib.h"
#include "../../header/dvs_fifo.h"
#include "../../header/dvs_thread.h"
#include "../../header/dvs_render.h"

#ifndef O_BINARY 
#  define O_BINARY 0
#endif

#if defined(__sun)
#  define PAGESIZE 0x2000
#elif defined(sgi)
#  define PAGESIZE 0x4000
#elif defined (__alpha__)
#  define PAGESIZE 0x2000
#elif defined(__ia64__)
#  define PAGESIZE 0x4000
#elif defined(__x86_64__)
#  define PAGESIZE 0x1000
#else
#  define PAGESIZE 0x1000
#endif

#define MAX_PATHNAME    512

#define MAX_BUFFERSIZE  (4096 * 3112 * 4)


#if defined(WIN32) || defined(linux)
#define fromle(x)                    (x)
#else
#define fromle(x)                    ((((x)&0xff000000)>>24)|(((x)&0x00ff0000)>>8)|(((x)&0x0000ff00)<<8)|(((x)&0x000000ff)<<24))
#endif


/*************************** GENERIC ****************************/


typedef struct {
  sv_handle * pSv;
  sv_render_handle * pSvRh;

  int loopMode;
  int verbose;
  int threadCount;
  int startFrame;
  int frameCount;
  int bRetry;

  struct {
    int enable;
    int scalerType;
    struct {
      double start;
      double end;
    } xsize,ysize,xoffset,yoffset,sharpness;
  } scale;

  struct {
    int enable;
    int valueSet;
    struct {
      int enable;
      double start;
      double end;
    } factorRed,factorGreen,factorBlue;
  } matrix;

  struct {
    void * pLut;   // Pointer to the buffer containing the 3D LUT data
    int    size;   // Size of the LUT data, e.g. BGRA * 4 bytes * entries -> 4 * 4 * 1024 for a 10-bit 1D LUT
    int    enable;
  } lut1d;

  struct {
    void * pLut;   // Pointer to the buffer containing the 3D LUT data
    int    size;   // Size of the LUT data, e.g. BGR * 2 bytes * entries -> 3 * 2 * (17*17*17) for a 16-bit 3D LUT
    int    enable;
  } lut3d;

  struct {
    int xsize;
    int ysize;
  } input,output;

  struct {
    char   fileName[MAX_PATHNAME];
    int    frameNr;
    char * allocPointer;
    char * pointer;
    void * handle;
    uint64 position;
  } source,destination;
} renderHandle;

#define ProgramError()      printf("ERROR: program error %s/%d\n", __FILE__, __LINE__); running = FALSE;

#include "../common/dpxformat.h"

